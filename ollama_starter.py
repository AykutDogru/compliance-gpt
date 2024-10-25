# ollama_starter.py

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from qdrant_client import QdrantClient  
from langchain_qdrant import QdrantVectorStore
from qdrant_client.http.models import Distance, VectorParams
from pypdf import PdfReader
from datetime import datetime
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import JsonOutputParser
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from graph_state import GraphState
from langgraph.graph import END, StateGraph, START
from pprint import pprint


class OllamaStarter:

    def __init__(self, local_llm,top_p,top_k,temperature, collection_name, uploaded_files,question):
        self.local_llm = local_llm
        self.collection_name = collection_name
        self.uploaded_files = uploaded_files
        self.question = question
        self.top_p = top_p
        self.top_k = top_k
        self.temperature = temperature
        self.qdrant_client = QdrantClient("http://172.28.0.3:6333")
        self.embedding = FastEmbedEmbeddings()
        self.doc_splits = []
        self.retriever = None
        self.app = None

    def load_documents(self):
        docs = []
        try:
            for file in self.uploaded_files:
                if file.name.endswith('.pdf'):
                    text = self.read_pdf(file)
                    upload_date = datetime.now().isoformat()
                    docs.append(Document(page_content=text, metadata={"filename": file.name, "upload_date": upload_date}))
            return docs, "Documents loaded successfully."
        except Exception as e:
            return [], f"Error loading documents: {str(e)}"

    def read_pdf(self, file):
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text

    def split_documents(self, docs_list):
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=1000, chunk_overlap=200
        )
        return text_splitter.split_documents(docs_list), "Documents split successfully."

    def get_existing_filenames(self):
        # Retrieve all documents and their metadata
        existing_docs = self.qdrant_client.scroll(collection_name=self.collection_name, limit=1000)

        # Assuming existing_docs is a tuple, access the documents correctly
        documents = existing_docs[0]  # Adjust this index based on your actual data structure

        # Extract  filename
        results = []
        for doc in documents:
            payload = doc.payload 
            results.append(payload['metadata']['filename'])
        return results


    def initialize_vector_store(self):
        collections = self.qdrant_client.get_collections()
        collection_names = [col.name for col in collections.collections]

        if self.collection_name not in collection_names:
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE),
                timeout=300
            )
            vectorstore = QdrantVectorStore.from_documents(
                documents=self.doc_splits,
                collection_name=self.collection_name,
                embedding=self.embedding
            )
            self.retriever = vectorstore.as_retriever()
            return vectorstore, f"Created new collection: {self.collection_name}"
        else:
            vectorstore = QdrantVectorStore(client=self.qdrant_client, collection_name=self.collection_name, embedding=self.embedding)
            return vectorstore, f"Using existing collection: {self.collection_name}"
        
    def connect_existing_vector_store(self):
        collections = self.qdrant_client.get_collections()
        collection_names = [col.name for col in collections.collections]
        vector_database_exists = False 
        if self.collection_name in collection_names:
            vectorstore = QdrantVectorStore(client=self.qdrant_client, collection_name=self.collection_name, embedding=self.embedding)
            self.retriever = vectorstore.as_retriever()
            vector_database_exists = True 
            return vector_database_exists,f"Using vector store with existing collection: {self.collection_name}"
        else:
            return vector_database_exists,f"Vector store does not exists with collection: {self.collection_name}"

    def process(self):
        docs, load_message = self.load_documents()
        if not docs:
            return load_message, None

        self.doc_splits, split_message = self.split_documents(docs)
        vectorstore, init_message = self.initialize_vector_store()
        
        # Get existing document filenames
        existing_filenames = self.get_existing_filenames()


        # Filter out new documents
        new_documents = [doc for doc in self.doc_splits if doc.metadata['filename'] not in existing_filenames]

        # Add new documents to the vector store
        if new_documents:
            vectorstore.add_documents(new_documents)

        added_count = len(new_documents)
        return f"{load_message} {split_message} {init_message} Added {added_count} new documents to {self.collection_name}.", vectorstore

    def get_document_summary(self):
        return f"Processed {len(self.doc_splits)} document chunks for collection: {self.collection_name}"
    

    # Post-processing
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
        
    def answer_question(self,user_question):
        self.question=user_question
        # LLM
        llm = ChatOllama(model=self.local_llm, format="json",base_url="http://172.28.0.2:11434",top_k=self.top_k,top_p=self.top_p,temperature=self.temperature)

        prompt = PromptTemplate(
        template="""You are a grader assessing relevance of a retrieved document to a user question. \n 
        Here is the retrieved document: \n\n {document} \n\n
        Here is the user question: {question} \n
        If the document contains keywords related to the user question, grade it as relevant. \n
        It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
        Provide the binary score as a JSON with a single key 'score' and no premable or explanation.""",
        input_variables=["question", "document"],
        )

        self.retrieval_grader = prompt | llm | JsonOutputParser()
        #question = user_question
        #docs = self.retriever.invoke(question)
        #doc_txt = docs[1].page_content
        #print(self.retrieval_grader.invoke({"question": question, "document": doc_txt}))

        ### Generate

        # Prompt
        prompt = hub.pull("rlm/rag-prompt")

        # LLM
        llm = ChatOllama(model=self.local_llm,base_url="http://172.28.0.2:11434",top_k=self.top_k,top_p=self.top_p,temperature=self.temperature)


        # Chain
        self.rag_chain = prompt | llm | StrOutputParser()

        # Run
        #self.generation = self.rag_chain.invoke({"context": docs, "question": question})
        #print(self.generation)



        ### Hallucination Grader

        # LLM
        llm = ChatOllama(model=self.local_llm, format="json",base_url="http://172.28.0.2:11434",top_k=self.top_k,top_p=self.top_p,temperature=self.temperature)

        # Prompt
        prompt = PromptTemplate(
            template="""You are a grader assessing whether an answer is grounded in / supported by a set of facts. \n 
            Here are the facts:
            \n ------- \n
            {documents} 
            \n ------- \n
            Here is the answer: {generation}
            Give a binary score 'yes' or 'no' score to indicate whether the answer is grounded in / supported by a set of facts. \n
            Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.""",
            input_variables=["generation", "documents"],
        )

        self.hallucination_grader = prompt | llm | JsonOutputParser()
        #self.hallucination_grader.invoke({"documents": docs, "generation": self.generation})


        ### Answer Grader

        # LLM
        llm = ChatOllama(model=self.local_llm, format="json",base_url="http://172.28.0.2:11434",top_k=self.top_k,top_p=self.top_p,temperature=self.temperature)
 
        # Prompt
        prompt = PromptTemplate(
            template="""You are a grader assessing whether an answer is useful to resolve a question. \n 
            Here is the answer:
            \n ------- \n
            {generation} 
            \n ------- \n
            Here is the question: {question}
            Give a binary score 'yes' or 'no' to indicate whether the answer is useful to resolve a question. \n
            Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.""",
            input_variables=["generation", "question"],
        )

        self.answer_grader = prompt | llm | JsonOutputParser()
        #self.answer_grader.invoke({"question": question, "generation": self.generation})


        ### Question Re-writer

        # LLM
        llm = ChatOllama(model=self.local_llm,base_url="http://172.28.0.2:11434",top_k=self.top_k,top_p=self.top_p,temperature=self.temperature)

        # Prompt
        re_write_prompt = PromptTemplate(
            template="""You a question re-writer that converts an input question to a better version that is optimized \n 
            for vectorstore retrieval. Look at the initial and formulate an improved question. \n
            Here is the initial question: \n\n {question}. Improved question with no preamble: \n """,
            input_variables=["generation", "question"],
        )

        self.question_rewriter = re_write_prompt | llm | StrOutputParser()
        #self.question_rewriter.invoke({"question": question})
        return self.build_graph()



### Nodes


    def retrieve(self,state):
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
        documents = self.retriever.invoke(question)
        return {"documents": documents, "question": question}


    def generate(self,state):
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
        print("question:",question)
        print("documents:",documents)
        # RAG generation
        generation = self.rag_chain.invoke({"context": documents, "question": question})
        return {"documents": documents, "question": question, "generation": generation}


    def grade_documents(self,state):
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
            score = self.retrieval_grader.invoke(
                {"question": question, "document": d.page_content}
            )
            grade = score["score"]
            print("grade_documents_question:",question)
            print("grade_documents_score:",score)
            if grade == "yes":
                print("---GRADE: DOCUMENT RELEVANT---")
                filtered_docs.append(d)
            else:
                print("---GRADE: DOCUMENT NOT RELEVANT---")
                continue
        return {"documents": filtered_docs, "question": question}


    def transform_query(self,state):
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
        better_question = self.question_rewriter.invoke({"question": question})
        return {"documents": documents, "question": better_question}


    ### Edges


    def decide_to_generate(self,state):
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


    def grade_generation_v_documents_and_question(self,state):
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

        score = self.hallucination_grader.invoke(
            {"documents": documents, "generation": generation}
        )
        grade = score["score"]
        print("grade_hallucination_question:",question)
        print("grade_hallucination_documents:",documents)
        print("grade_hallucination_generation:",generation)
        print("grade_hallucination_score:",score)
        # Check hallucination
        if grade == "yes":
            print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
            # Check question-answering
            print("---GRADE GENERATION vs QUESTION---")
            score = self.answer_grader.invoke({"question": question, "generation": generation})
            grade = score["score"]
            print("answer_grader_score:",grade)
            if grade == "yes":
                print("---DECISION: GENERATION ADDRESSES QUESTION---")
                return "useful"
            else:
                print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
                return "not useful"
        else:
            print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
            return "not supported"
        



    def build_graph(self):
        workflow = StateGraph(GraphState)

        # Define the nodes
        workflow.add_node("retrieve", self.retrieve)  # retrieve
        workflow.add_node("grade_documents", self.grade_documents)  # grade documents
        workflow.add_node("generate", self.generate)  # generatae
        workflow.add_node("transform_query", self.transform_query)  # transform_query

        # Build graph
        workflow.add_edge(START, "retrieve")
        workflow.add_edge("retrieve", "grade_documents")
        workflow.add_conditional_edges(
            "grade_documents",
            self.decide_to_generate,
            {
                "transform_query": "transform_query",
                "generate": "generate",
            },
        )
        workflow.add_edge("transform_query", "retrieve")
        workflow.add_conditional_edges(
            "generate",
            self.grade_generation_v_documents_and_question,
            {
                "not supported": "generate",
                "useful": END,
                "not useful": "transform_query",
            },
        )

        # Compile
        self.app = workflow.compile()
        return self.execute_graph()



    def execute_graph(self):
        # Run
        inputs = {"question": self.question}
        print("inputs:",inputs)
        for output in self.app.stream(inputs):
            for key, value in output.items():
                # Node
                pprint(f"Node '{key}':")
                # Optional: print full state at each node
                # pprint.pprint(value["keys"], indent=2, width=80, depth=None)
            pprint("\n---\n")

        # Final generation
        return value["generation"]




