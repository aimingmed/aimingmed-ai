import json
import os
import argparse
import shutil

from decouple import config
from typing import List
from typing_extensions import TypedDict

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from langchain_deepseek import ChatDeepSeek
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.tools.tavily_search import TavilySearchResults

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate, HumanMessagePromptTemplate

from langchain.schema import Document
from pprint import pprint
from langgraph.graph import END, StateGraph, START

from models.adaptive_rag.routing import RouteQuery
from models.adaptive_rag.grading import (
    GradeDocuments,
    GradeHallucinations,
    GradeAnswer,
)
from models.adaptive_rag.query import (
    QueryRequest,
    QueryResponse,
)

from models.adaptive_rag.prompts_library import (
    system_router,
    system_retriever_grader,
    system_hallucination_grader,
    system_answer_grader,
    system_question_rewriter,
    qa_prompt_template
)

from .utils import ConnectionManager

router = APIRouter()

# Load environment variables
os.environ["DEEPSEEK_API_KEY"] = config(
    "DEEPSEEK_API_KEY", cast=str, default="sk-XXXXXXXXXX"
)
os.environ["TAVILY_API_KEY"] = config(
    "TAVILY_API_KEY", cast=str, default="tvly-dev-wXXXXXX"
)

# Initialize embedding model (do this ONCE)
embedding_model = HuggingFaceEmbeddings(model_name="paraphrase-multilingual-mpnet-base-v2")

# Initialize the DeepSeek chat model
llm = ChatDeepSeek(
    model="deepseek-chat",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

# Load data from ChromaDB
db_folder = "chroma_db"
db_path = os.path.join(os.getcwd(), db_folder)
collection_name = "rag-chroma"
vectorstore = Chroma(persist_directory=db_path, collection_name=collection_name, embedding_function=embedding_model)
retriever = vectorstore.as_retriever()


############################ LLM functions ############################
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

### Generate
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

############### Graph functions ################

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

# Initialize the connection manager
manager = ConnectionManager()

@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()

            try:
                data_json = json.loads(data)
                if (
                    isinstance(data_json, list)
                    and len(data_json) > 0
                    and "content" in data_json[0]
                ):  
                    inputs = {
                        "question": data_json[0]["content"]
                    }
                    async for chunk in app.astream(inputs):
                        # Determine if chunk is intermediate or final
                        if isinstance(chunk, dict):
                            if len(chunk) == 1:
                                step_name = list(chunk.keys())[0]
                                step_value = chunk[step_name]
                                # Check if this step contains the final answer
                                if isinstance(step_value, dict) and 'generation' in step_value:
                                    await manager.send_personal_message(
                                        json.dumps({
                                            "type": "final",
                                            "title": "Answer",
                                            "payload": step_value['generation']
                                        }),
                                        websocket,
                                    )
                                else:
                                    await manager.send_personal_message(
                                        json.dumps({
                                            "type": "intermediate",
                                            "title": step_name.replace('_', ' ').title(),
                                            "payload": str(step_value)
                                        }),
                                        websocket,
                                    )
                            elif 'generation' in chunk:
                                await manager.send_personal_message(
                                    json.dumps({
                                        "type": "final",
                                        "title": "Answer",
                                        "payload": chunk['generation']
                                    }),
                                    websocket,
                                )
                            else:
                                await manager.send_personal_message(
                                    json.dumps({
                                        "type": "intermediate",
                                        "title": "Step",
                                        "payload": str(chunk)[:500]
                                    }),
                                    websocket,
                                )
                        else:
                            # Fallback for non-dict chunks
                            await manager.send_personal_message(
                                json.dumps({
                                    "type": "intermediate",
                                    "title": "Step",
                                    "payload": str(chunk)[:500]
                                }),
                                websocket,
                            )
                    # Send a final 'done' message to signal completion
                    await manager.send_personal_message(
                        json.dumps({"type": "done"}),
                        websocket,
                    )
                else:
                    await manager.send_personal_message(
                        "Invalid message format", websocket
                    )

            except json.JSONDecodeError:
                await manager.broadcast("Invalid JSON message")

    except WebSocketDisconnect:
        manager.disconnect(websocket)
        await manager.broadcast("Client disconnected")

    except WebSocketDisconnect:
        manager.disconnect(websocket)
        await manager.broadcast("Client disconnected")
