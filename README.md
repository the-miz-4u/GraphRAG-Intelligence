# 🕸️ GraphRAG Intelligence System

An advanced, offline-first Retrieval-Augmented Generation (RAG) system powered by Knowledge Graphs. This project extracts entities and relationships from various unstructured data sources to build a dynamic knowledge network, allowing users to query information with high accuracy, conversational memory, and verifiable citations.

## ✨ Key Features

* **Multi-Modal Data Ingestion:**
  * 📄 **PDF Uploads:** Automatically extracts text and knowledge from PDF documents using `PyPDF2`.
  * ✍️ **Raw Text Input:** Manually paste text for quick processing.
  * 🌐 **Web Scraping:** Enter any Wikipedia or blog URL to automatically scrape and process web content using `BeautifulSoup`.
* **Automated Knowledge Graph Construction:** Utilizes **Llama 3.2** (running locally via Ollama) to intelligently extract Entities and Relationships, automatically formatting and saving them into a **Neo4j** database.
* **Conversational AI with Memory:** A sleek chat interface that remembers the context of the conversation (last 5 interactions) to handle pronoun resolutions (e.g., "he", "it") accurately.
* **Verifiable Citations:** Eliminates AI hallucinations by strictly grounding answers in the Neo4j graph and providing explicit source citations for every fact.
* **Interactive Data Visualization:** Explore your data visually with a physics-based, interactive network graph powered by **Pyvis**.
* **Smart Database Management:** Includes a dedicated "Danger Zone" in the sidebar to format/clear the entire graph and chat history with a single click, keeping the environment clean for new topics.
* **100% Local & Private:** Designed to run entirely on local infrastructure with zero data leaving your machine.

## 🛠️ Tech Stack

* **Frontend UI:** Streamlit
* **LLM Engine:** Ollama (Llama 3.2)
* **Graph Database:** Neo4j
* **Data Processing & Parsing:** LangChain, PyPDF2, BeautifulSoup4, Requests
* **Graph Visualization:** Pyvis, HTML/JS
* **Language:** Python 3

## 🚀 Installation & Setup

### Prerequisites
1. **Python 3.8+** installed on your system.
2. **Ollama:** Download and install [Ollama](https://ollama.com/), then pull the Llama 3.2 model:
   ```bash
   ollama run llama3.2

Neo4j Desktop: Install Neo4j Desktop, create a local DBMS, set the password to password (or update it in app.py), and start the database on port 7687.
Running the Application
Clone the repository:

Bash
git clone [https://github.com/the-miz-4u/GraphRAG-Intelligence.git](https://github.com/the-miz-4u/GraphRAG-Intelligence.git)
cd GraphRAG-Intelligence


2. **Create and activate a virtual environment (Recommended):**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Mac/Linux:
   source venv/bin/activate
Install dependencies:

Bash
pip install streamlit pyvis langchain_community neo4j PyPDF2 requests beautifulsoup4


4. **Start the Streamlit App:**
   ```bash
   streamlit run app.py

   💡 How to Use
Ingest Knowledge: Open the app and use the left panel to upload a PDF, paste text, or enter a Web URL. Click "Process" to let the AI build the graph.

Visualize: Navigate to the "Interactive Graph" tab on the right to see how the AI has connected your data points. You can drag the nodes around!

Chat: Go to the "Chat with Data" tab and ask questions. The AI will traverse the graph to answer and provide exact citations.

Manage Data: Use the "Format / Clear Entire Graph" button in the sidebar when you want to wipe the database and start fresh with a new document or topic.

Author:

Manish Sharma | B.Tech Computer Science and Engineering

Passionate about AI, Machine Learning, and Full-Stack Engineering.
