## TASK 1: LLM-Powered AI Prototype

# 📄 PDF AI Assistant

A full-stack application that allows users to upload PDF documents and interact with them using natural language. Built with **FastAPI**, **React**, **LangChain**, and **Hugging Face**.

## 🚀 Features

- **Deployed & Live**: Fully hosted on Hugging Face Spaces (Backend) and GitHub Pages (Frontend).

- **PDF Upload & Processing**: Automatically extracts text, chunks it, and generates vector embeddings.
- **RAG Pipeline**: Uses Retrieval-Augmented Generation to answer questions based *only* on the PDF content.
- **LLM Integration**: Powered by **Meta LLaMA 3.1 8B Instruct** (via Hugging Face API) for high-quality responses.
- **Vector Search**: Efficient similarity search using **FAISS** (with in-memory fallback).
- **Interactive Chat UI**:
  - Real-time chat interface.
  - **Markdown Support**: Renders lists, code blocks, and bold text properly.
  - **Auto-scroll**: Keeps the latest message in view.
  - **Persistence**: Chat history is saved locally so you don't lose progress on refresh.
- **Responsive Design**: Beautiful, modern UI with gradient styling.

### 🚀 Live Demo 👉 https://llm-powered-automating-pdf-interaction.onrender.com

## 🛠️ Tech Stack

### Backend
- **Framework**: FastAPI
- **Hosting**: Hugging Face Spaces (Docker)
- **LLM Orchestration**: LangChain
- **Model Provider**: Hugging Face Inference API
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` (Local)
- **Vector Store**: FAISS (Facebook AI Similarity Search)

### Frontend
- **Library**: React.js
- **Hosting**: GitHub Pages
- **Styling**: CSS3 (Custom gradients & animations)
- **HTTP Client**: Axios
- **Rendering**: `react-markdown`

## 📋 Prerequisites

- Python 3.8+
- Node.js & npm
- A [Hugging Face Account](https://huggingface.co/) & API Token.
- Access to `meta-llama/Meta-Llama-3.1-8B-Instruct` (Accept license on HF model page).

## ⚙️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/divyansshu/Automating-PDF-Interaction.git
cd "Automating PDF Interaction"
```

### 2. Backend Setup
Navigate to the backend folder and install dependencies:
```bash
cd backend
# Create virtual environment (optional but recommended)
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

**Configure Environment Variables:**
Create a `.env` file in `backend/app/.env`:
```env
HUGGINGFACEHUB_API_TOKEN=your_hf_token_here
```

**Run the Server:**
```bash
cd app
uvicorn main:app --reload
```
The backend will start at `http://localhost:8000`.

### 3. Frontend Setup
Open a new terminal and navigate to the frontend folder:
```bash
cd frontend
npm install
npm start
```
The app will open at `http://localhost:3000`.

## 💡 Usage

1.  Open the web app.
2.  Click **"Upload PDF"** and select a document.
3.  Wait for the processing (extraction & embedding generation).
4.  Once done, you will be redirected to the chat screen.
5.  Ask any question about your PDF!
6.  Use the **"Upload New PDF"** button to reset and start over.

## 🛡️ License

This project is open-source and available under the MIT License.

## 🔄 Workflow

Think of the system as a **conversation pipeline** where your PDF transforms into an interactive knowledge base. Here’s how the magic happens:

1. 📤 **Upload PDF** → User uploads a document.  
2. 📑 **Text Extraction** → Content is extracted using `pdfplumber` / `PyPDF2`.  
3. ✂️ **Chunking** → Text is split into overlapping segments (~1000 characters, 200 overlap) to preserve context.  
4. 🧠 **Embeddings** → Each chunk is converted into dense vectors using `OpenAIEmbeddings` or `SentenceTransformers`.  
5. 🗂️ **Vector Database** → Embeddings are stored in **FAISS** (local) or **Pinecone** (scalable cloud).  
6. ❓ **User Query** → Query is embedded and matched against the most relevant chunks.  
7. 📝 **Prompt Engineering** → Retrieved context is inserted into a carefully designed prompt.  
8. 🤖 **LLM Response** → GPT (or LLaMA2/Mistral) generates an answer grounded in the PDF.  
9. 💬 **UI Display** → Streamlit chat interface shows the response with conversational flow.  

---

## 🎨 Design Choices

- **LLM**:  
  - 🟢 *OpenAI GPT* → Reliable, strong reasoning.  
  - 🟡 *LLaMA2/Mistral* → Cost-effective, privacy-friendly alternatives.  

- **Vector DB**:  
  - ⚡ *FAISS* → Fast, local prototyping.  
  - ☁️ *Pinecone* → Enterprise scaling with metadata filtering.  

- **Chunking Strategy**:  
  - Sliding window with overlap ensures no sentence or context is cut off mid-thought.  

- **Prompt Engineering**:  
  - Source-grounded prompts reduce hallucination.  
  - Explicit instructions: *“If unsure, say ‘Not found in document.’”*  

- **UI**:  
  - 🚀 *Streamlit* → Rapid prototyping with minimal setup.  
  - 🔗 *FastAPI* → Optional REST endpoints for production integration.  

---

## 📦 Requirements

##Add the following dependencies to `requirements.txt`:

```txt
streamlit
langchain
openai
faiss-cpu
PyPDF2
pdfplumber

### Project Structure

├── app.py              # Streamlit/FastAPI entry point
├── requirements.txt    # Dependencies
├── README.md           # Documentation
├── /docs               # Architecture diagrams, notes
└── /src                # Core code (chunking, embeddings, RAG pipeline)

## Add Architecture Diagram
[PDF Upload] → [Chunking] → [Embeddings → Vector DB] → [Retriever] → [LLM] → [UI]

## 🛡️ Task 2: Hallucination & Quality Control

### Causes of Hallucination
- LLMs may generate confident but incorrect answers when:
  - Context is missing or incomplete.
  - Retrieval returns irrelevant chunks.
  - Prompts are ambiguous or unconstrained.

### Guardrails Implemented
1. **Confidence Thresholds**  
   - Responses are only generated if similarity score > threshold.  
   - Otherwise, the system replies: *“Not found in document.”*

2. **Source-Grounded Answers**  
   - All answers are explicitly tied to retrieved chunks.  
   - Prompt constraint: *“Use only the provided context. Do not invent information.”*

3. **Prompt Constraints**  
   - Instructions force the model to avoid speculation.  
   - Example: *“If unsure, say ‘Not found in document.’”*

### Example of Improved Responses
- **Before (Hallucination)**:  
  *“The company was founded in 1990.”*  
- **After (Guardrail Applied)**:  
  *“The founding year is not mentioned in the document. Closest reference is early operations.”*

---

## ⚡ Task 3: Rapid Iteration Challenge

### Advanced Capability: Multi-Document Reasoning
**Why chosen**: Real-world use cases often involve multiple PDFs (contracts, resumes, reports).  
**Implementation**:  
- Ingest multiple PDFs → Merge embeddings into one vector DB.  
- Retrieval → Query across all documents simultaneously.  
- Prompt → Include source identifiers (e.g., Doc A, Doc B).  

**Trade-offs**:
- ✅ Richer, enterprise-ready answers.  
- ❌ Higher compute cost and retrieval complexity.  
- 🔒 Limitation: Requires metadata filtering for relevance.

---

## 🏢 Task 4: AI System Architecture

### Enterprise Assistant Design

**Components:**
- **Data Ingestion**: ETL pipeline (PDFs, docs, emails → text).  
- **Vector DB Choice**: Pinecone/Weaviate for scalability and metadata filtering.  
- **LLM Orchestration**: LangChain/LlamaIndex for RAG pipeline management.  
- **Cost Control**:  
  - Cache embeddings.  
  - Use smaller LLM for retrieval, larger LLM for final answer.  
- **Monitoring & Evaluation**:  
  - Track query success rate.  
  - Log hallucinations.  
  - Human feedback loop for continuous improvement.

### Architecture Diagram
[Data Sources: PDFs, Docs, Emails]
        ↓
 [ETL + Chunking]
        ↓
 [Embeddings → Vector DB (Pinecone)]
        ↓
 [Retriever → Top-k Chunks]
        ↓
 [LLM Orchestration (LangChain)]
        ↓
 [Response Generation + Guardrails]
        ↓
 [UI Layer (Streamlit / FastAPI)]
        ↓
 [Monitoring + Feedback Loop]

### Project Structure
├── app.py              # Streamlit app entry point
├── requirements.txt    # Dependencies
├── README.md           # Documentation
├── /docs               # Architecture diagrams, notes
└── /src                # Core code (chunking, embeddings, RAG pipeline)

---

## ✨ Features
- LLM-powered PDF Q&A  
- RAG with FAISS  
- Chunking strategy  
- Guardrails against hallucination  
- Multi-document reasoning  
- Enterprise-ready architecture  

---

## 📝 Author
**Anmol Tomar** – MCA AIML, 590019134  
Focus: LLMs, RAG, NLP, and enterprise AI systems.
