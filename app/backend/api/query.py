import os
import logging
import argparse

from decouple import config
from langchain_deepseek import ChatDeepSeek
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from fastapi import FastAPI, APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any
from typing_extensions import TypedDict

from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.prompts import PromptTemplate, HumanMessagePromptTemplate

from langchain.schema import Document
from pprint import pprint
from langgraph.graph import END, StateGraph, START

from app.backend.models.adaptive_rag.data_models import (
    RouteQuery, 
    GradeDocuments, 
    GradeHallucinations, 
    GradeAnswer
)
from app.backend.models.adaptive_rag.prompts_library import (
    system_router, 
    system_retriever_grader,
    system_hallucination_grader,
    system_answer_grader,
    system_question_rewriter,
    qa_prompt_template
)


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

os.environ["DEEPSEEK_API_KEY"] = config("DEEPSEEK_API_KEY", cast=str)
os.environ["TAVILY_API_KEY"] = config("TAVILY_API_KEY", cast=str)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

app = FastAPI()
router = APIRouter()

class QueryRequest(BaseModel):
    query: str = Field(..., description="The question to ask the model")
    input_chromadb_artifact: str = Field(..., description="Fully-qualified name for the chromadb artifact")
    embedding_model: str = Field("paraphrase-multilingual-mpnet-base-v2", description="Sentence Transformer model name")
    chat_model_provider: str = Field("gemini", description="Chat model provider")


class QueryResponse(BaseModel):
    response: str = Field(..., description="The model's response")


@router.post("/query", response_model=QueryResponse, response_model_exclude_none=True)
async def query_endpoint(request: Request, query_request: QueryRequest):
    try:
        args = argparse.Namespace(
            query=query_request.query,
            input_chromadb_artifact=query_request.input_chromadb_artifact,
            embedding_model=query_request.embedding_model,
            chat_model_provider=query_request.chat_model_provider
        )
        result = go(args)
        return {"response": result["response"]}
    except Exception as e:
        logger.exception(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {e}")


def go(args):

        logger.info("Downloading chromadb artifact")
        artifact_chromadb_local_path = args.input_chromadb_artifact #modified
        # shutil.unpack_archive(artifact_chromadb_local_path, "chroma_db") #removed

        # Initialize embedding model (do this ONCE)
        embedding_model = HuggingFaceEmbeddings(model_name=args.embedding_model) 
        llm = ChatDeepSeek(
                model="deepseek-chat", 
                temperature=0,
                max_tokens=None,
                timeout=None,
                max_retries=2,
        )

        # Load data from ChromaDB
        # db_folder = "chroma_db" #removed
        # db_path = os.path.join(os.getcwd(), db_folder) #removed
        # collection_name = "rag-chroma" #removed
        vectorstore = Chroma(persist_directory=artifact_chromadb_local_path, collection_name="rag-chroma", embedding_function=embedding_model) #modified
        retriever = vectorstore.as_retriever()

        ##########################################
        # Routing to vectorstore or web search
        structured_llm_router = llm.with_structured_output(RouteQuery)
        # Prompt
        route_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_router),
                ("human", "{question}"),
            ]
        )
        question_router = route_prompt | structured_llm_router

        ##########################################
        ### Retrieval Grader
        structured_llm_grader = llm.with_structured_output(GradeDocuments)
        # Prompt
        grade_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_retriever_grader),
                ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
            ]
        )
        retrieval_grader = grade_prompt | structured_llm_grader

        ##########################################
        ### Generate
        from langchain_core.output_parsers import StrOutputParser

        # Create a PromptTemplate with the given prompt
        new_prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template=qa_prompt_template,
        )

        # Create a new HumanMessagePromptTemplate with the new PromptTemplate
        new_human_message_prompt_template = HumanMessagePromptTemplate(
            prompt=new_prompt_template
        )
        prompt_qa = ChatPromptTemplate.from_messages([new_human_message_prompt_template])

        # Chain
        rag_chain = prompt_qa | llm | StrOutputParser()


        ##########################################
        ### Hallucination Grader
        structured_llm_grader = llm.with_structured_output(GradeHallucinations)

        # Prompt
        hallucination_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_hallucination_grader),
                ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
            ]
        )

        hallucination_grader = hallucination_prompt | structured_llm_grader

        ##########################################
        ### Answer Grader
        structured_llm_grader = llm.with_structured_output(GradeAnswer)

        # Prompt
        answer_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_answer_grader),
                ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
            ]
        )
        answer_grader = answer_prompt | structured_llm_grader

        ##########################################
        ### Question Re-writer
        # Prompt
        re_write_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_question_rewriter),
                (
                    "human",
                    "Here is the initial question: \n\n {question} \n Formulate an improved question.",
                ),
            ]
        )   
        question_rewriter = re_write_prompt | llm | StrOutputParser()

        ### Search
        web_search_tool = TavilySearchResults(k=3)

        class GraphState(TypedDict):
            """
            Represents the state of our graph.

            Attributes:
                question: question
                generation: LLM generation
                documents: list of documents
            """

            question: str
            generation: str
            documents: List[str]



        def retrieve(state):
            """
            Retrieve documents

            Args:
                state (dict): The current graph state

            Returns:
                state (dict): New key added to state, documents, that contains retrieved documents
            """
            print("---RETRIEVE---")
            question = state["question"]

            # Retrieval
            documents = retriever.invoke(question)

            print(documents)
            return {"documents": documents, "question": question}


        def generate(state):
            """
            Generate answer

            Args:
                state (dict): The current graph state

            Returns:
                state (dict): New key added to state, generation, that contains LLM generation
            """
            print("---GENERATE---")
            question = state["question"]
            documents = state["documents"]

            # RAG generation
            generation = rag_chain.invoke({"context": documents, "question": question})
            return {"documents": documents, "question": question, "generation": generation}


        def grade_documents(state):
            """
            Determines whether the retrieved documents are relevant to the question.

            Args:
                state (dict): The current graph state

            Returns:
                state (dict): Updates documents key with only filtered relevant documents
            """

            print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
            question = state["question"]
            documents = state["documents"]

            # Score each doc
            filtered_docs = []
            for d in documents:
                score = retrieval_grader.invoke(
                    {"question": question, "document": d.page_content}
                )
                grade = score.binary_score
                if grade == "yes":
                    print("---GRADE: DOCUMENT RELEVANT---")
                    filtered_docs.append(d)
                else:
                    print("---GRADE: DOCUMENT NOT RELEVANT---")
                    continue
            return {"documents": filtered_docs, "question": question}


        def transform_query(state):
            """
            Transform the query to produce a better question.

            Args:
                state (dict): The current graph state

            Returns:
                state (dict): Updates question key with a re-phrased question
            """

            print("---TRANSFORM QUERY---")
            question = state["question"]
            documents = state["documents"]

            # Re-write question
            better_question = question_rewriter.invoke({"question": question})
            return {"documents": documents, "question": better_question}


        def web_search(state):
            """
            Web search based on the re-phrased question.

            Args:
                state (dict): The current graph state

            Returns:
                state (dict): Updates documents key with appended web results
            """

            print("---WEB SEARCH---")
            question = state["question"]

            # Web search
            docs = web_search_tool.invoke({"query": question})
            web_results = "\n".join([d["content"] for d in docs])
            web_results = Document(page_content=web_results)

            return {"documents": web_results, "question": question}


        ### Edges ###
        def route_question(state):
            """
            Route question to web search or RAG.

            Args:
                state (dict): The current graph state

            Returns:
                str: Next node to call
            """

            print("---ROUTE QUESTION---")
            question = state["question"]
            source = question_router.invoke({"question": question})
            if source.datasource == "web_search":
                print("---ROUTE QUESTION TO WEB SEARCH---")
                return "web_search"
            elif source.datasource == "vectorstore":
                print("---ROUTE QUESTION TO RAG---")
                return "vectorstore"


        def decide_to_generate(state):
            """
            Determines whether to generate an answer, or re-generate a question.

            Args:
                state (dict): The current graph state

            Returns:
                str: Binary decision for next node to call
            """

            print("---ASSESS GRADED DOCUMENTS---")
            state["question"]
            filtered_documents = state["documents"]

            if not filtered_documents:
                # All documents have been filtered check_relevance
                # We will re-generate a new query
                print(
                    "---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---"
                )
                return "transform_query"
            else:
                # We have relevant documents, so generate answer
                print("---DECISION: GENERATE---")
                return "generate"


        def grade_generation_v_documents_and_question(state):
            """
            Determines whether the generation is grounded in the document and answers question.

            Args:
                state (dict): The current graph state

            Returns:
                str: Decision for next node to call
            """

            print("---CHECK HALLUCINATIONS---")
            question = state["question"]
            documents = state["documents"]
            generation = state["generation"]

            score = hallucination_grader.invoke(
                {"documents": documents, "generation": generation}
            )
            grade = score.binary_score

            # Check hallucination
            if grade == "yes":
                print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
                # Check question-answering
                print("---GRADE GENERATION vs QUESTION---")
                score = answer_grader.invoke({"question": question, "generation": generation})
                grade = score.binary_score
                if grade == "yes":
                    print("---DECISION: GENERATION ADDRESSES QUESTION---")
                    return "useful"
                else:
                    print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
                    return "not useful"
            else:
                pprint("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
                return "not supported"

        workflow = StateGraph(GraphState)

        # Define the nodes
        workflow.add_node("web_search", web_search)  # web search
        workflow.add_node("retrieve", retrieve)  # retrieve
        workflow.add_node("grade_documents", grade_documents)  # grade documents
        workflow.add_node("generate", generate)  # generatae
        workflow.add_node("transform_query", transform_query)  # transform_query

        # Build graph
        workflow.add_conditional_edges(
            START,
            route_question,
            {
                "web_search": "web_search",
                "vectorstore": "retrieve",
            },
        )
        workflow.add_edge("web_search", "generate")
        workflow.add_edge("retrieve", "grade_documents")
        workflow.add_conditional_edges(
            "grade_documents",
            decide_to_generate,
            {
                "transform_query": "transform_query",
                "generate": "generate",
            },
        )
        workflow.add_edge("transform_query", "retrieve")
        workflow.add_conditional_edges(
            "generate",
            grade_generation_v_documents_and_question,
            {
                "not supported": "generate",
                "useful": END,
                "not useful": "transform_query",
            },
        )

        # Compile
        app = workflow.compile()

        # Run
        inputs = {
            "question": args.query
        }
        for output in app.stream(inputs):
            for key, value in output.items():
                # Node
                pprint(f"Node '{key}':")
                # Optional: print full state at each node
                # pprint.pprint(value["keys"], indent=2, width=80, depth=None)
            pprint("\n---\n")

        # Final generation
        print(value["generation"])

        return {"response": value["generation"]}
    


app.include_router(router, prefix="/query", tags=["query"])
