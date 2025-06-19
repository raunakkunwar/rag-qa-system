import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.document_loaders import TextLoader

# --- Config ---
HUGGINGFACEHUB_API_TOKEN = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
if not HUGGINGFACEHUB_API_TOKEN:
    print("Warning: Hugging Face API token not found. Set the HUGGINGFACEHUB_API_TOKEN environment variable.")

KNOWLEDGE_BASE_PATH = "knowledge_base.txt"
MODEL_REPO_ID = "google/flan-t5-base" # FLAN-T5 is a solid choice for this

def setup_rag_chain():
    """Build the RAG chain."""
    # 1. Load the document
    print("Loading knowledge base...")
    if not os.path.exists(KNOWLEDGE_BASE_PATH):
        raise FileNotFoundError(f"'{KNOWLEDGE_BASE_PATH}' not found. Please create it.")

    loader = TextLoader(KNOWLEDGE_BASE_PATH)
    documents = loader.load()

    # 2. Split it into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.split_documents(documents)
    print(f"Split document into {len(docs)} chunks.")

    # 3. Create embeddings
    # This runs locally on the CPU
    print("Creating text embeddings...")
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )

    # 4. Create a vector store with the embeddings
    print("Creating vector store...")
    vector_store = FAISS.from_documents(docs, embedding_model)
    # This retriever will find the relevant docs
    retriever = vector_store.as_retriever()

    # 5. Get the LLM for the generation step
    print("Loading LLM for generation...")
    llm = HuggingFaceHub(
        repo_id=MODEL_REPO_ID,
        model_kwargs={"temperature": 0.2, "max_length": 512}
    )

    # 6. Create a prompt to guide the LLM
    template = """
    Use the context below to answer the question.
    If you don't know the answer, just say that you don't know.

    Context: {context}
    Question: {question}

    Answer:
    """
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])

    # 7. Tie it all together in a chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff", # "stuff" puts all context into one prompt
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    return qa_chain

def main():
    """Runs the main interactive Q&A loop."""
    print("Setting up the RAG chain...")
    try:
        qa_chain = setup_rag_chain()
    except Exception as e:
        print(f"Error during setup: {e}")
        return

    print("\n> RAG chain is ready. Ask a question or type 'exit' to quit.")
    while True:
        user_question = input("\nYour question: ")
        if user_question.lower() == 'exit':
            print("> Goodbye!")
            break

        print("> Thinking...")
        try:
            result = qa_chain({"query": user_question})
            print(f"\n> {result['result'].strip()}")
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
