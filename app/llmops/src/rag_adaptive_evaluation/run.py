import os
import logging
import argparse
import mlflow
import shutil
import langsmith

from decouple import config
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_deepseek import ChatDeepSeek
from langchain_community.llms.moonshot import Moonshot
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.chroma import Chroma


from typing import List
from typing_extensions import TypedDict

from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.prompts import PromptTemplate, HumanMessagePromptTemplate

from langchain.schema import Document
from pprint import pprint
from langgraph.graph import END, StateGraph, START
from langsmith import Client


from data_models import (
    RouteQuery, 
    GradeDocuments, 
    GradeHallucinations, 
    GradeAnswer
)
from prompts_library import (
    system_router, 
    system_retriever_grader,
    system_hallucination_grader,
    system_answer_grader,
    system_question_rewriter,
    qa_prompt_template
)

from evaluators import (
    gemini_evaluator_correctness,
    deepseek_evaluator_correctness,
    moonshot_evaluator_correctness,
    gemini_evaluator_faithfulness,
    deepseek_evaluator_faithfulness,
    moonshot_evaluator_faithfulness
)

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

os.environ["GOOGLE_API_KEY"] = config("GOOGLE_API_KEY", cast=str)
os.environ["DEEPSEEK_API_KEY"] = config("DEEPSEEK_API_KEY", cast=str)
os.environ["MOONSHOT_API_KEY"] = config("MOONSHOT_API_KEY", cast=str)
os.environ["TAVILY_API_KEY"] = config("TAVILY_API_KEY", cast=str)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["LANGSMITH_API_KEY"] = config("LANGSMITH_API_KEY", cast=str)
os.environ["LANGSMITH_TRACING"] = config("LANGSMITH_TRACING", cast=str)
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGSMITH_PROJECT"] = config("LANGSMITH_PROJECT", cast=str)

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
            )
        elif args.chat_model_provider == 'gemini':
            llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash", 
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
            )

        # Load data from ChromaDB
        db_folder = "chroma_db"
        db_path = os.path.join(os.getcwd(), db_folder)
        collection_name = "rag-chroma"
        vectorstore = Chroma(persist_directory=db_path, collection_name=collection_name, embedding_function=embedding_model)
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
    
def go_evaluation(args):
    if args.evaluation_dataset_csv_path:

        import pandas as pd

        df = pd.read_csv(args.evaluation_dataset_csv_path)
        dataset_name = os.path.basename(args.evaluation_dataset_csv_path).split('.')[0]

        # df contains columns of question and answer
        examples = df[[args.evaluation_dataset_column_question, args.evaluation_dataset_column_answer]].values.tolist()
        inputs = [{"question": input_prompt} for input_prompt, _ in examples]
        outputs = [{"answer": output_answer} for _, output_answer in examples]

        # Programmatically create a dataset in LangSmith
        client = Client()

        try:
            # Create a dataset
            dataset = client.create_dataset(
                dataset_name = dataset_name,
                description = "An evaluation dataset in LangSmith."
            )
            # Add examples to the dataset
            client.create_examples(inputs=inputs, outputs=outputs, dataset_id=dataset.id)
        except langsmith.utils.LangSmithConflictError:
            pass

        
        args.ls_chat_model_evaluator = None if args.ls_chat_model_evaluator == 'None' else args.ls_chat_model_evaluator.split(',')

        def target(inputs: dict) -> dict:
            new_args = argparse.Namespace(**vars(args))
            new_args.query = inputs["question"]
            return go(new_args)

        ls_evaluators = []
        if args.ls_chat_model_evaluator:
            for evaluator in args.ls_chat_model_evaluator:
                if evaluator == 'moonshot':
                    ls_evaluators.append(moonshot_evaluator_correctness)
                    ls_evaluators.append(moonshot_evaluator_faithfulness)
                elif evaluator == 'deepseek':
                    ls_evaluators.append(deepseek_evaluator_correctness)
                    ls_evaluators.append(deepseek_evaluator_faithfulness)
                elif evaluator == 'gemini':
                    ls_evaluators.append(gemini_evaluator_correctness)
                    ls_evaluators.append(gemini_evaluator_faithfulness)
        
        # After running the evaluation, a link will be provided to view the results in langsmith
        _ = client.evaluate(
            target,
            data = dataset_name,
            evaluators = ls_evaluators,
            experiment_prefix = "first-eval-in-langsmith",
            max_concurrency = 1,
            
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adaptive AG")

    parser.add_argument(
        "--query", 
        type=str,
        help="Question to ask the model",
        required=True
    )

    parser.add_argument(
        "--evaluation_dataset_csv_path",
        type=str,
        help="Path to the query evaluation dataset",
        default=None,
    )

    parser.add_argument(
        "--evaluation_dataset_column_question",
        type=str,
        help="Column name for the questions in the evaluation dataset",
        default="question",
    )

    parser.add_argument(
        "--evaluation_dataset_column_answer",
        type=str,
        help="Column name for the groundtruth answers in the evaluation dataset",
        default="groundtruth",
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

    parser.add_argument(
        "--ls_chat_model_evaluator",
        type=str,
        help="list of Chat model providers for evaluation",
        required=False,
        default="None"
    )

    args = parser.parse_args()

    go_evaluation(args)
