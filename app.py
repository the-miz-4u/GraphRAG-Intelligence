import streamlit as st
from langchain_community.graphs import Neo4jGraph
from langchain_community.llms import Ollama

# ==========================================
# COMPONENT 1: KnowledgeGraphManager
# ==========================================
class KnowledgeGraphManager:
    def __init__(self, uri, username, password):
        self.graph = Neo4jGraph(
            url=uri, 
            username=username, 
            password=password, 
            refresh_schema=False
        )
        
    def add_relationship(self, entity1, relation, entity2, source):
        rel_safe = relation.upper().replace(' ', '_')
        query = f"""
        MERGE (n1:Entity {{name: $e1}})
        MERGE (n2:Entity {{name: $e2}})
        MERGE (n1)-[r:{rel_safe} {{source: $source}}]->(n2)
        """
        self.graph.query(query, params={"e1": entity1, "e2": entity2, "source": source})
        
    def get_context(self, keywords):
        # SMART SEARCH: Search for multiple keywords at once
        query = """
        UNWIND $keywords AS kw
        MATCH (n:Entity)-[r]-(m:Entity) 
        WHERE toLower(n.name) CONTAINS toLower(kw)
        RETURN DISTINCT n.name, type(r), m.name, r.source LIMIT 20
        """
        return self.graph.query(query, params={"keywords": keywords})

# ==========================================
# COMPONENT 2: Extract Entities using LLM
# ==========================================
def extract_entities_with_llm(text, llm):
    prompt = f"""
    Extract all entities and their relationships from the text below. 
    Format strictly as: Entity1 | Relationship | Entity2
    Example: Manish | STUDIES AT | UEM Jaipur
    
    Text: {text}
    """
    response = llm.invoke(prompt)
    relationships = []
    
    for line in response.split('\n'):
        if '|' in line:
            parts = [p.strip() for p in line.split('|')]
            if len(parts) == 3:
                relationships.append(parts)
    return relationships

# ==========================================
# COMPONENT 3: RAG with Citations
# ==========================================
def generate_answer_with_citations(question, kg_manager, llm):
    # SMART KEYWORD EXTRACTION:
    # Filter out common stop words and keep important entities
    stop_words = {'is', 'there', 'any', 'connection', 'between', 'and', 'the', 'who', 'what', 'where', 'how', 'in', 'a', 'of', 'to'}
    words = question.lower().replace('?', '').split()
    search_keywords = [w for w in words if w not in stop_words]
    
    if not search_keywords:
        search_keywords = words # Fallback

    context_data = kg_manager.get_context(search_keywords)
    
    if not context_data:
        return "I couldn't find any relevant information in the graph database.", []

    context_text = ""
    citations = []
    
    for i, data in enumerate(context_data):
        fact = f"{data['n.name']} {data['type(r)']} {data['m.name']}"
        context_text += f"[{i+1}] {fact}\n"
        citations.append(f"[{i+1}] Source Fact: {fact}")

    prompt = f"""
    You are a Knowledge Graph Assistant. 
    Answer the question based ONLY on the provided graph context.
    Use [1], [2] to cite your sources. 
    If multiple facts connect entities (like a chain), explain the connection.

    Context:
    {context_text}
    
    Question: {question}
    Answer:
    """
    answer = llm.invoke(prompt)
    return answer, citations


# ==========================================
# UI CODE: Streamlit Frontend
# ==========================================
st.set_page_config(page_title="GraphRAG Intelligence", layout="wide")
st.title("🕸️ Knowledge Graph RAG System")

st.sidebar.header("System Status")

try:
    llm = Ollama(model="llama3.2")
    st.sidebar.success("✅ Llama 3.2: Online")
except:
    st.sidebar.error("❌ Ollama: Offline")

try:
    # Use 127.0.0.1 for local stability
    kg_manager = KnowledgeGraphManager("neo4j://127.0.0.1:7687", "neo4j", "password")
    st.sidebar.success("✅ Neo4j: Connected")
except Exception as e:
    st.sidebar.error(f"❌ Connection Error: {e}")

col1, col2 = st.columns([1, 1.5])

with col1:
    st.header("1. Knowledge Ingestion")
    doc_input = st.text_area("Paste source text:", height=250)
    
    if st.button("Build Knowledge Graph"):
        if doc_input:
            with st.spinner("Analyzing and Mapping..."):
                extracted = extract_entities_with_llm(doc_input, llm)
                for rel in extracted:
                    kg_manager.add_relationship(rel[0], rel[1], rel[2], "User_Upload")
                st.success(f"Successfully mapped {len(extracted)} relationships!")
        else:
            st.warning("Please provide input text.")

with col2:
    st.header("2. Graph-Augmented Query")
    query = st.chat_input("Ask about Manish, TechNova, or Mark Zuckerberg...")
    
    if query:
        st.chat_message("user").write(query)
        with st.spinner("Traversing graph nodes..."):
            ans, cites = generate_answer_with_citations(query, kg_manager, llm)
            with st.chat_message("assistant"):
                st.write(ans)
                if cites:
                    st.divider()
                    st.markdown("#### 📚 Verifiable Citations")
                    for c in cites:
                        st.caption(c)