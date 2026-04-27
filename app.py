import PyPDF2
from pyvis.network import Network
import streamlit.components.v1 as components
import streamlit as st
from langchain_community.graphs import Neo4jGraph
from langchain_community.llms import Ollama

# ==========================================
# COMPONENT 1: KnowledgeGraphManager  
# ==========================================

#ye def pdf se text nikal raha h
def extract_text_from_pdf(uploaded_file):
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        if page.extract_text():
            text += page.extract_text() + "\n"
    return text

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
    def get_all_data_for_visuals(self):
        # UI mein dikhane ke liye database se saare relationships nikalna
        query = """
        MATCH (n:Entity)-[r]->(m:Entity) 
        RETURN n.name AS source, type(r) AS relation, m.name AS target LIMIT 100
        """
        return self.graph.query(query)
    
    def build_graph_from_text(self, text, llm):
        # Yeh prompt LLM ko batayega ki entities aur relationships kaise nikalni hai
        prompt = f"""
        Extract key entities and their relationships from the following text.
        Format: Entity1 | Relationship | Entity2
        Keep it concise.
        Text: {text}
        """
        response = llm.invoke(prompt)
        
        # Response ko lines mein todna aur database mein save karna
        lines = response.split('\n')
        for line in lines:
            if "|" in line:
                parts = line.split("|")
                if len(parts) == 3:
                    e1 = parts[0].strip()
                    rel = parts[1].strip()
                    e2 = parts[2].strip()
                    # Relationships save karna
                    self.add_relationship(e1, rel, e2, "Uploaded Document")
    
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
def generate_answer_with_citations(question, kg_manager, llm, history):
    # History ko ek string mein badalna taaki LLM ko bhej sakein
    chat_history_text = "\n".join([f"{m['role']}: {m['content']}" for m in history[-5:]]) # Last 5 baatein

    # Baki keyword extraction logic same rahega
    stop_words = {'is', 'there', 'any', 'connection', 'between', 'and', 'the', 'who', 'what', 'where', 'how', 'in', 'a', 'of', 'to'}
    words = question.lower().replace('?', '').split()
    search_keywords = [w for w in words if w not in stop_words]
    
    context_data = kg_manager.get_context(search_keywords)
    
    if not context_data:
        context_text = "No direct graph facts found for these keywords."
    else:
        context_text = ""
        for i, data in enumerate(context_data):
            context_text += f"[{i+1}] {data['n.name']} {data['type(r)']} {data['m.name']}\n"

    # Naya Prompt jo Memory ko samajhta hai
    prompt = f"""
    You are a Knowledge Graph Assistant with Memory.
    
    Recent Chat History:
    {chat_history_text}
    
    Current Graph Context:
    {context_text}
    
    User Question: {question}
    
    Instruction: Use the Chat History to understand pronouns like 'he', 'it', or 'they'. 
    Answer based on the Graph Context. If the answer is in history but not in context, you can refer to it.
    Answer:
    """
    answer = llm.invoke(prompt)
    
    citations = [f"Source Fact: {data['n.name']} {data['type(r)']} {data['m.name']}" for data in (context_data or [])]
    return answer, citations
# ==========================================
# COMPONENT 4: Interactive Visual Graph
# ==========================================
def render_interactive_graph(kg_manager):
    data = kg_manager.get_all_data_for_visuals()
    
    if not data:
        st.info("Graph is empty. Please upload some text first to build the network.")
        return

    # Streamlit dark theme ke hisaab se graph design karna
    net = Network(height="450px", width="100%", bgcolor="#0E1117", font_color="white", directed=True)

    for record in data:
        source = record['source']
        target = record['target']
        relation = record['relation']

        # Nodes aur Edges add karna (Colors Streamlit theme se match kiye hain)
        net.add_node(source, label=source, color="#FF4B4B")
        net.add_node(target, label=target, color="#008CC1")
        net.add_edge(source, target, title=relation, label=relation, color="#A0AEC0")

    # Physics on karna taaki nodes aapas mein repel karein (bouncy effect)
    net.repulsion(node_distance=150, spring_length=200)

    try:
        # HTML file generate karke Streamlit mein dikhana
        net.save_graph("graph.html")
        HtmlFile = open("graph.html", 'r', encoding='utf-8')
        source_code = HtmlFile.read()
        components.html(source_code, height=500)
    except Exception as e:
        st.error(f"Visualizer Error: {e}")

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
st.sidebar.divider() # Ek line draw karne ke liye
if st.sidebar.button("🗑️ Clear Chat History"):
    st.session_state.messages = []
    st.rerun()

col1, col2 = st.columns([1, 1.5])

with col1:
    st.header("1. Knowledge Ingestion")
    
    # Do tabs: Ek upload ke liye, ek copy-paste ke liye
    ingest_tab1, ingest_tab2 = st.tabs(["📄 Upload Document", "✍️ Paste Text"])
    
    with ingest_tab1:
        st.markdown("Upload a PDF or Text file to automatically extract knowledge.")
        uploaded_file = st.file_uploader("Choose a file", type=["pdf", "txt"])
        
        if uploaded_file is not None:
            if st.button("Process Document"):
                with st.spinner("Reading file and building Knowledge Graph... This may take a while!"):
                    try:
                        # File read karna (PDF ya TXT)
                        if uploaded_file.name.endswith('.pdf'):
                            source_text = extract_text_from_pdf(uploaded_file)
                        else:
                            source_text = uploaded_file.read().decode("utf-8")
                            
                        # Graph banana (Yahan llm pass karna zaroori hai)
                        kg_manager.build_graph_from_text(source_text, llm)
                        st.success(f"Graph successfully built from {uploaded_file.name}!")
                    except Exception as e:
                        st.error(f"Error processing file: {e}")

    with ingest_tab2:
        source_text = st.text_area("Paste source text:", height=250)
        if st.button("Build Knowledge Graph"):
            if source_text:
                with st.spinner("Extracting entities and relationships..."):
                    # Yahan bhi llm pass kijiye
                    kg_manager.build_graph_from_text(source_text, llm)
                    st.success("Graph built successfully!")
            else:
                st.warning("Please enter some text first.")

                
with col2:
    st.header("2. AI Query & Visualization")
    
    # Do tabs banayein
    tab1, tab2 = st.tabs(["💬 Chat with Data", "🕸️ Interactive Graph"])
    
    with tab1:
        # 1. Initialize Chat History agar pehle se nahi hai
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # 2. Purane saare messages screen par dikhana
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # 3. Naya user input lena
        if query := st.chat_input("Ask me anything..."):
            # User message dikhana aur save karna
            st.chat_message("user").markdown(query)
            st.session_state.messages.append({"role": "user", "content": query})

            with st.spinner("Thinking..."):
                # Memory (st.session_state.messages) ke saath answer generate karna
                ans, cites = generate_answer_with_citations(query, kg_manager, llm, st.session_state.messages)
                
                # Assistant response dikhana aur save karna
                with st.chat_message("assistant"):
                    st.markdown(ans)
                    if cites:
                        with st.expander("View Graph Sources"):
                            for c in cites:
                                st.caption(c)
                
                st.session_state.messages.append({"role": "assistant", "content": ans})
                            
    with tab2:
        st.markdown("### Your Knowledge Network")
        st.caption("Drag the nodes to interact with your data!")
        # Humara naya function call karna
        render_interactive_graph(kg_manager)