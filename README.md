# GraphRAG Intelligence System

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Neo4j](https://img.shields.io/badge/Neo4j-008CC1?style=for-the-badge&logo=neo4j&logoColor=white)
![Ollama](https://img.shields.io/badge/Ollama-White?style=for-the-badge&logo=ollama&logoColor=black)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)

An advanced, 100% offline **Retrieval-Augmented Generation (RAG)** system that uses a **Knowledge Graph** to connect the dots between entities, eliminating AI hallucinations and enabling multi-hop reasoning.

## 🚀 Key Features

- **Privacy-First & Offline:** Runs entirely on your local machine. No data is sent to external APIs.
- **Deep Relationship Mapping:** Uses Neo4j to store data as interconnected nodes, allowing the AI to understand indirect relationships (A -> B -> C).
- **Smart Context Extraction:** Powered by **Llama 3.2** (via Ollama) to accurately extract entities and relationships from unstructured text.
- **Verifiable Citations:** The AI doesn't just answer; it provides exact source facts from the graph database to prove its logic.

## 🧠 System Architecture

1. **Ingestion:** Text -> LLM Entity Extraction -> Neo4j Graph Creation
2. **Retrieval:** User Query -> Smart Keyword Search -> Graph Traversal
3. **Generation:** Graph Context -> Local LLM -> Reasoned Output with Citations

## 📸 Snapshot
<img width="1919" height="1079" alt="image" src="https://github.com/user-attachments/assets/5a8e928c-b216-4598-a5cf-cc17d59a0cf7" />

## 🛠️ Prerequisites
- [Docker Desktop](https://www.docker.com/) (For running Neo4j)
- [Ollama](https://ollama.com/) (For running Llama 3.2 locally)
- Python 3.9+

## 🏁 Quick Start Guide

**1. Start the Graph Database (Neo4j)**
Run the Neo4j container using Docker:
```bash
docker start neo4j
```

**2. Start the AI Model (Ollama)**
Ensure Ollama is running in the background with the Llama 3.2 model pulled.

**3. Setup the Application**
Clone the repository and install the dependencies:
```bash
git clone [https://github.com/the-miz-4u/GraphRAG-Intelligence.git](https://github.com/the-miz-4u/GraphRAG-Intelligence.git)
cd GraphRAG-Intelligence
pip install -r requirements.txt
```

**4. Launch the App**
```bash
streamlit run app.py
```

## 👨‍💻 Developed By
**Manish Sharma** 
