# LLM-Powered Q&A with Retrieval-Augmented Generation (RAG)

This project implements a Question-Answering system using the Retrieval-Augmented Generation (RAG) pattern. It demonstrates how to ground a Large Language Model (LLM) on a custom knowledge base, enabling it to answer specific questions about information it wasn't originally trained on.

Instead of a simple classification task, this system understands a user's query, retrieves relevant information from a provided text file, and then uses a generative LLM to synthesize a natural language answer based on that context. This is a powerful, real-world technique used to build specialized chatbots and knowledge retrieval systems.

### Key Features
- **Retrieval-Augmented Generation (RAG):** The core of the project, combining information retrieval with generative AI.
- **Custom Knowledge Base:** Uses a simple text file (`knowledge_base.txt`) as the source of truth.
- **Vector Embeddings:** Employs `sentence-transformers` to create meaningful numerical representations of text.
- **Vector Store:** Utilizes `FAISS` (Facebook AI Similarity Search) for efficient, in-memory similarity searches.
- **LLM for Generation:** Uses a pre-trained generative model (`google/flan-t5-base`) from Hugging Face to generate answers.
- **Orchestration:** Leverages the `langchain` library to streamline the entire RAG pipeline.

### Technologies Used
- **Python 3.8+**
- **PyTorch**
- **Hugging Face Transformers**
- **LangChain**
- **Sentence-Transformers**
- **FAISS (CPU version)**

---

### Setup and Installation

Follow these steps to set up the project environment.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/raunakkunwar/rag-qa-system.git
    cd rag-qa-system
    ```

2.  **Create your knowledge base:**
    Create a file named `knowledge_base.txt` in the root directory and add some text to it.

3.  **Create and activate a virtual environment (recommended):**
    ```bash
    # For Unix/macOS
    python3 -m venv venv
    source venv/bin/activate

    # For Windows
    python -m venv venv
    .\venv\Scripts\activate
    ```

4.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

---

### How to Run

1.  **Prepare your Knowledge Base:**
    Make sure the `knowledge_base.txt` file exists and has content. The repository includes an example about cricket.

2.  **Set your Hugging Face API Token:**
    Before running, set your API token as an environment variable. You can get a free token from `huggingface.co/settings/tokens`.
    ```bash
    # On macOS/Linux
    export HUGGINGFACEHUB_API_TOKEN='your_token_here'

    # On Windows (CMD)
    set HUGGINGFACEHUB_API_TOKEN=your_token_here
    ```

3.  **Run the application:**
    ```bash
    python run_rag_qa.py
    ```
    The script will set up the RAG chain and enter an interactive loop where you can ask questions about the content in `knowledge_base.txt`.

### Example Interaction
```
> Setting up the RAG chain. This may take a moment...
> RAG chain is ready. Ask a question or type 'exit' to quit.

Your question: What is Test cricket?
> Thinking...
> Test cricket is the traditional form of cricket, played over five days with no limit on the number of overs.

Your question: Which organization governs cricket?
> Thinking...
> The International Cricket Council (ICC) is the primary governing body for cricket.

Your question: exit
> Goodbye!
```
