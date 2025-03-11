import os
import logging
import argparse
import mlflow
import shutil
from decouple import config
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_deepseek import ChatDeepSeek
from langchain_community.llms.moonshot import Moonshot
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.vectorstores.chroma import Chroma

from typing import Literal, List
from typing_extensions import TypedDict

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.schema import Document
from pprint import pprint
from langgraph.graph import END, StateGraph, START

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

GEMINI_API_KEY = config("GOOGLE_API_KEY", cast=str)
DEEKSEEK_API_KEY = config("DEEKSEEK_API_KEY", cast=str)
MOONSHOT_API_KEY = config("MOONSHOT_API_KEY", cast=str)
TAVILY_API_KEY = config("TAVILY_API_KEY", cast=str)
os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def go(args):

    # start a new MLflow run
    with mlflow.start_run(experiment_id=mlflow.get_experiment_by_name("development").experiment_id, run_name="etl_chromdb_pdf"):
        existing_params = mlflow.get_run(mlflow.active_run().info.run_id).data.params
        if 'query' not in existing_params:
            mlflow.log_param('query', args.query)
        
        # Log parameters to MLflow
        mlflow.log_params({
            "input_chromadb_artifact": args.input_chromadb_artifact,
            "embedding_model": args.embedding_model,
            "chat_model_provider": args.chat_model_provider
        })


        logger.info("Downloading chromadb artifact")
        artifact_chromadb_local_path = mlflow.artifacts.download_artifacts(artifact_uri=args.input_chromadb_artifact)

        # unzip the artifact
        logger.info("Unzipping the artifact")
        shutil.unpack_archive(artifact_chromadb_local_path, "chroma_db")

        # Initialize embedding model (do this ONCE)
        embedding_model = HuggingFaceEmbeddings(model_name=args.embedding_model) 
        if args.chat_model_provider == 'deepseek':
            llm = ChatDeepSeek(
                model="deepseek-chat", 
                temperature=0,
                max_tokens=None,
                timeout=None,
                max_retries=2,
                api_key=DEEKSEEK_API_KEY
            )
        elif args.chat_model_provider == 'gemini':
            llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash", 
                google_api_key=GEMINI_API_KEY,
                temperature=0,
                max_retries=3,
                streaming=True
            )
        elif args.chat_model_provider == 'moonshot':
            llm = Moonshot(
                model="moonshot-v1-128k", 
                temperature=0,
                max_tokens=None,
                timeout=None,
                max_retries=2,
                api_key=MOONSHOT_API_KEY
            )

        # Load data from ChromaDB
        db_folder = "chroma_db"
        db_path = os.path.join(os.getcwd(), db_folder)
        collection_name = "rag-chroma"
        vectorstore = Chroma(persist_directory=db_path, collection_name=collection_name, embedding_function=embedding_model)
        retriever = vectorstore.as_retriever()

        # Data model
        class RouteQuery(BaseModel):
            """Route a user query to the most relevant datasource."""

            datasource: Literal["vectorstore", "web_search"] = Field(
                ...,
                description="Given a user question choose to route it to web search or a vectorstore.",
            )

        structured_llm_router = llm.with_structured_output(RouteQuery)

        # Prompt
        system = """You are an expert at routing a user question to a vectorstore or web search.
        The vectorstore contains documents related to medical treatment for cancer/tumor diseases.
        Use the vectorstore for questions on these topics. Otherwise, use web-search."""
        route_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", "{question}"),
            ]
        )

        question_router = route_prompt | structured_llm_router


        ### Retrieval Grader
        # Data model
        class GradeDocuments(BaseModel):
            """Binary score for relevance check on retrieved documents."""

            binary_score: str = Field(
                description="Documents are relevant to the question, 'yes' or 'no'"
            )

        structured_llm_grader = llm.with_structured_output(GradeDocuments)

        # Prompt
        system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
            If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
            It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
            Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
        grade_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
            ]
        )

        retrieval_grader = grade_prompt | structured_llm_grader


        ### Generate

        from langchain import hub
        from langchain_core.output_parsers import StrOutputParser

        # Prompt
        prompt = hub.pull("rlm/rag-prompt")

        # Post-processing
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)


        # Chain
        rag_chain = prompt | llm | StrOutputParser()




        ### Hallucination Grader

        # Data model
        class GradeHallucinations(BaseModel):
            """Binary score for hallucination present in generation answer."""

            binary_score: str = Field(
                description="Answer is grounded in the facts, 'yes' or 'no'"
            )


        # LLM with function call
        structured_llm_grader = llm.with_structured_output(GradeHallucinations)

        # Prompt
        system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
            Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""
        hallucination_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
            ]
        )

        hallucination_grader = hallucination_prompt | structured_llm_grader


        ### Answer Grader
        # Data model
        class GradeAnswer(BaseModel):
            """Binary score to assess answer addresses question."""

            binary_score: str = Field(
                description="Answer addresses the question, 'yes' or 'no'"
            )


        # LLM with function call
        structured_llm_grader = llm.with_structured_output(GradeAnswer)

        # Prompt
        system = """You are a grader assessing whether an answer addresses / resolves a question \n 
            Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question."""
        answer_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
            ]
        )

        answer_grader = answer_prompt | structured_llm_grader

        ### Question Re-writer

        # LLM

        # Prompt
        system = """You a question re-writer that converts an input question to a better version that is optimized \n 
            for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning."""
        re_write_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
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
        pprint(value["generation"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chain of Thought RAG")

    parser.add_argument(
        "--query", 
        type=str,
        help="Question to ask the model",
        required=True
    )

    parser.add_argument(
        "--input_chromadb_artifact", 
        type=str,
        help="Fully-qualified name for the chromadb artifact",
        required=True
    )

    parser.add_argument(
        "--embedding_model",
        type=str,
        default="paraphrase-multilingual-mpnet-base-v2",
        help="Sentence Transformer model name"
    )

    parser.add_argument(
        "--chat_model_provider",
        type=str,
        default="gemini",
        help="Chat model provider"
    )

    args = parser.parse_args()
    
    go(args)