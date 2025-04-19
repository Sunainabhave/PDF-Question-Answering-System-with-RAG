import os
import streamlit as st
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from qdrant_client.models import PointStruct
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
import google.generativeai as genai
from dotenv import load_dotenv

# ===== Load Environment Variables =====
load_dotenv()

# ===== Configuration =====
# Initialize Qdrant client
qdrant_client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)

# Initialize sentence transformer model for embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load the Gemini API Key from the environment
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel('gemini-1.5-flash')

# Constants
COLLECTION_NAME = "pdf_collection"
VECTOR_SIZE = 384
CHUNK_SIZE = 1000

# ===== Core Functions =====
def extract_text_from_pdf(pdf_file):
    """Extract text from PDF"""
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def chunk_text(text):
    """Chunk text into smaller pieces"""
    chunks = []
    for i in range(0, len(text), CHUNK_SIZE):
        chunk = text[i:i + CHUNK_SIZE]
        if chunk.strip():
            chunks.append(chunk)
    return chunks

def get_embeddings(chunks):
    """Generate embeddings for the text chunks"""
    return model.encode(chunks)

def setup_collection():
    """Set up the Qdrant collection for storing the embeddings"""
    try:
        qdrant_client.delete_collection(COLLECTION_NAME)
    except:
        pass
    qdrant_client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=VECTOR_SIZE,
            distance=Distance.COSINE
        )
    )

def upload_to_qdrant(chunks, embeddings):
    """Upload text chunks and embeddings to Qdrant"""
    points = [
        PointStruct(id=i, vector=embedding, payload={"text": chunk})
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings))
    ]
    qdrant_client.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )

def search_chunks(query_vector):
    """Search for the most relevant chunks in Qdrant"""
    result = qdrant_client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=5  # Number of chunks to retrieve
    )
    return [hit.payload["text"] for hit in result]

def generate_answer_from_gemini(query, context):
    """Generate an answer from the Google Gemini API"""
    prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
    try:
        response = gemini_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Error generating answer from Gemini: {e}"

# ===== Streamlit UI =====
st.set_page_config(layout="wide")
st.title("📄 PDF Question Answering System with RAG")

# File upload and question input
pdf = st.file_uploader("Upload PDF", type="pdf")
question = st.text_input("Ask a question about the PDF:")

# Initialize variables
answer = ""
if pdf:
    # Save the uploaded PDF to a temporary location
    with open("uploaded_pdf.pdf", "wb") as f:
        f.write(pdf.read())

    # Extract text and chunk it
    text = extract_text_from_pdf("uploaded_pdf.pdf")
    chunks = chunk_text(text)
    embeddings = get_embeddings(chunks)

    # Set up collection and upload to Qdrant
    setup_collection()
    upload_to_qdrant(chunks, embeddings)

if question:
    # Generate embeddings for the question
    query_vector = model.encode([question])[0]

    # Retrieve relevant context chunks from Qdrant
    context_chunks = search_chunks(query_vector)
    context = " ".join(context_chunks)

    # Generate an answer from Gemini
    answer = generate_answer_from_gemini(question, context)

# Display answer
if answer:
    st.subheader("Answer:")
    st.write(answer)