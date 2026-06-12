<div align="center">

# 🕸️ GraphRAG Intelligence System
*An Advanced, Offline-First Knowledge Graph & RAG Architecture*

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![Neo4j](https://img.shields.io/badge/Neo4j-018bff?logo=neo4j&logoColor=white)](https://neo4j.com/)
[![Ollama](https://img.shields.io/badge/LLM-Llama_3.2-orange.svg)](https://ollama.com/)

</div>

<br>

> **GraphRAG Intelligence** extracts entities and relationships from unstructured data (PDFs, Texts, URLs) to build a dynamic knowledge network. It enables users to query complex information with high accuracy, conversational memory, and verifiable citations.

---

## 📸 System Interface

<img width="1918" height="1078" alt="image" src="https://github.com/user-attachments/assets/a8501b3a-43ce-41a1-80f3-a29c3e1829b4" />


---

## ✨ Enterprise-Grade Features

* 🧠 **Automated Graph Construction:** Utilizes **Llama 3.2** (via Ollama) to extract `Entity-Relationship-Entity` triples and autonomously saves them into a **Neo4j** database.
* 📂 **Multi-Modal Data Ingestion:**
  * **PDF Uploads:** Automated extraction via `PyPDF2`.
  * **Web Scraping:** Fetch and parse live content from URLs using `BeautifulSoup`.
* 💬 **Context-Aware Memory:** A ChatGPT-style interface that retains recent chat history for seamless multi-turn conversations and accurate pronoun resolution.
* 🎯 **Verifiable Citations:** Eliminates AI hallucinations by grounding answers directly in the Graph DB and explicitly citing source nodes.
* 🕸️ **Interactive Visualization:** Explore data visually through a physics-based, draggable network graph powered by **Pyvis**.
* 🛡️ **100% Local & Private:** Zero data leaves your machine. Fully offline execution.

---

## 📸 Chat & Knowledge Querying
<img width="1918" height="1078" alt="image" src="https://github.com/user-attachments/assets/c42f1b40-f78e-41ca-bc41-2f46d4d07afe" />

---

## 🛠️ System Architecture & Tech Stack

| Component | Technology Used |
| :--- | :--- |
| **Frontend UI** | Streamlit |
| **LLM Engine** | Ollama (Llama 3.2) |
| **Graph Database** | Neo4j Desktop |
| **Parsing Engine** | PyPDF2, BeautifulSoup4, Requests |
| **Graph Visualization**| Pyvis, HTML/JavaScript |

---

## 🚀 Installation & Local Setup

### 1. Prerequisites
* **Ollama:** Install [Ollama](https://ollama.com/) and pull the local model:
  ```bash
  ollama run llama3.2
  
Neo4j: Install Neo4j Desktop, create a local DBMS, set password to password, and start it on port 7687.

2. Run the System
Bash
# Clone repository
git clone [https://github.com/the-miz-4u/GraphRAG-Intelligence.git](https://github.com/the-miz-4u/GraphRAG-Intelligence.git)
cd GraphRAG-Intelligence

# Create Virtual Environment
python -m venv venv
venv\Scripts\activate   # For Windows

# Install Dependencies
pip install streamlit pyvis langchain_community neo4j PyPDF2 requests beautifulsoup4

# Launch Application
streamlit run app.py
