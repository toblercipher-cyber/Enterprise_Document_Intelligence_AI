"""
Enterprise Document Intelligence — FastAPI Backend
====================================================
Wraps the LangGraph RAG workflow (resume / invoice / contract analyzer)
and exposes it as an HTTP API for the frontend.

Run:
    pip install fastapi uvicorn python-multipart pydantic \\
                langchain langchain-community langchain-huggingface \\
                langchain-groq langgraph pypdf faiss-cpu python-dotenv
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import operator
import shutil
import tempfile
from pathlib import Path
from typing import Annotated, Optional, TypedDict

from dotenv import load_dotenv

# ---- FastAPI ----
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ---- LangChain / RAG ----
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---- LangGraph ----
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

load_dotenv()

# ============================================================
# Configuration
# ============================================================
UPLOAD_DIR = Path(tempfile.gettempdir()) / "edi_uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # 20 MB

# ============================================================
# Models
# ============================================================
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.4)

# ============================================================
# LangGraph State
# ============================================================
class State(TypedDict):
    pdf_path: str
    extract_text: str
    document_type: str
    merge_result: Annotated[list, operator.add]
    document_text: str
    messages: Annotated[list, add_messages]
    final_report: str


NO_RETRIEVAL_NEEDED = "NO_RETRIEVAL_NEEDED"


# ============================================================
# Retriever builder
# ============================================================
def build_retriever(pdf_path: str):
    """Build a per-request retriever for the uploaded PDF."""
    loader = PyPDFLoader(pdf_path)
    document = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(document)

    vectorstore = FAISS.from_documents(chunks, embedding_model)
    return vectorstore.as_retriever(search_kwargs={"k": 4})


# ============================================================
# Nodes
# ============================================================
def extract_text_node(state: State) -> dict:
    loader = PyPDFLoader(state["pdf_path"])
    document = loader.load()
    full_text = "\n".join(page.page_content for page in document)
    return {"extract_text": full_text}


def classifier_node(state: State) -> dict:
    last_message = state["messages"][-1]
    extract_text = state["extract_text"]

    prompt = f"""You are a document classification assistant.
Analyze only the contents of the document text below and classify it into exactly one of these categories:
- resume
- invoice
- contract

User context (for reference only, do not classify based on this alone): {last_message.content}

Document text:
{extract_text}

Return only one word: resume, invoice, or contract.
Do not return explanations, punctuation, markdown, or any extra text."""

    response = llm.invoke(prompt)
    category = response.content.strip().lower()

    if category not in {"resume", "invoice", "contract"}:
        category = "unknown"

    return {"document_type": category}


def _run_retriever(state: State) -> dict:
    query = state["messages"][-1].content
    retriever = build_retriever(state["pdf_path"])
    docs = retriever.invoke(query)
    context = "\n\n".join(d.page_content for d in docs)
    return {"document_text": context}


def resume_analyzer_node(state: State) -> dict:
    return _run_retriever(state)


def invoice_analyzer_node(state: State) -> dict:
    return _run_retriever(state)


def contract_analyzer_node(state: State) -> dict:
    return _run_retriever(state)


def general_analyzer_node(state: State) -> dict:
    return {"document_text": NO_RETRIEVAL_NEEDED}


def route_function(state: State):
    if state["document_type"] == "resume":
        return "resume_analyzer_node"
    if state["document_type"] == "invoice":
        return "invoice_analyzer_node"
    if state["document_type"] == "contract":
        return "contract_analyzer_node"
    return "general_analyzer_node"


def response_node(state: State) -> dict:
    document_type = state["document_type"]
    context = state["document_text"]
    query = state["messages"][-1].content

    if context == NO_RETRIEVAL_NEEDED:
        prompt = f"""You are an expert assistant for {document_type} documents.

The detected document type is: {document_type}

Answer the user's question using your own knowledge. Tailor your answer according to the document type:
- resume -> answer as a resume expert
- invoice -> answer as an invoice/accounting expert
- contract -> answer as a legal contract expert

Handle requests such as rewriting, explanation, improvement, correction, extraction, summarization, or analysis as appropriate.

User question: {query}

Return only the final answer.
Do not mention retrieval, vector databases, or how the context was obtained."""
    else:
        prompt = f"""You are an expert assistant for {document_type} documents.

Detected document type: {document_type}

Retrieved document content:
{context}

User question: {query}

Base your answer primarily on the retrieved content above.
If the retrieved content is insufficient to answer confidently, clearly state what information is missing instead of guessing or hallucinating.
Adapt your response style according to the document type:
- resume -> career/resume assistant
- invoice -> financial/invoice assistant
- contract -> legal contract assistant"""

    result = llm.invoke(prompt)
    return {"messages": [("ai", result.content.strip())]}


def summary_node(state: State) -> dict:
    messages = state["messages"]
    prompt = f"""You are a highly skilled document summarization assistant with expertise in analyzing professional documents such as resumes, invoices, and contracts.

Your task is to carefully read and understand the content provided below, and then produce a high-quality summary.

Content to analyze:
{messages}

Instructions:
- Read the entire content carefully before summarizing.
- Identify the most important points, facts, and details — do not skip anything significant.
- Write the summary in clear, professional language that is easy to understand.
- Keep the summary concise, but make sure no critical information is lost.
- Do not add any assumptions, opinions, or information that is not explicitly present in the content.
- Organize the summary logically (e.g., in short paragraphs or bullet points, whichever suits the content best).
- If the content contains multiple sections or topics, make sure each one is represented in the summary.

Now generate the summary based on the above instructions."""
    result = llm.invoke(prompt)
    return {"merge_result": [f"Summary:\n{result.content.strip()}"]}


def risk_node(state: State) -> dict:
    messages = state["messages"]
    prompt = f"""You are an expert risk analysis assistant with deep experience in identifying potential issues, red flags, and concerns across different types of documents such as resumes, invoices, and contracts.

Your task is to carefully examine the content provided below and identify any risks, problems, or areas of concern.

Content to analyze:
{messages}

Instructions:
- Carefully read through the entire content before identifying risks.
- Look for issues such as: missing information, inconsistencies, unclear terms, financial discrepancies, legal exposure, unfavorable conditions, suspicious details, or anything that could negatively impact the reader.
- Think critically — consider not just what is written, but also what might be missing or unclear that should normally be present in this type of document.
- Explain each risk clearly and briefly, so the reader understands why it matters.
- Prioritize the risks if possible, starting with the most significant ones.
- If, after careful analysis, you find no significant risks, state that clearly and explain why the content appears sound.
- Do not fabricate risks that are not reasonably supported by the content.

Now generate a clear and well-reasoned risk analysis based on the above instructions."""
    result = llm.invoke(prompt)
    return {"merge_result": [f"Risks:\n{result.content.strip()}"]}


def keyword_node(state: State) -> dict:
    messages = state["messages"]
    prompt = f"""You are a highly capable keyword and entity extraction assistant, skilled at identifying the most meaningful and relevant terms from professional documents.

Your task is to carefully analyze the content provided below and extract the most important keywords, entities, and terms.

Content to analyze:
{messages}

Instructions:
- Read through the entire content carefully before extracting anything.
- Identify important elements such as: names of people or organizations, technical terms, skills, dates, monetary values, key topics, and any other significant entities relevant to the document.
- Focus on terms that are meaningful and would help someone quickly understand what the document is about, without needing to read it in full.
- Avoid including generic or filler words that add no real value.
- Present the extracted keywords and entities as a clear, organized list.
- If the content includes distinct categories of information (e.g., technical skills vs. names vs. dates), group them accordingly for clarity.

Now generate the extracted keywords and entities based on the above instructions."""
    result = llm.invoke(prompt)
    return {"merge_result": [f"Keywords:\n{result.content.strip()}"]}


def final_report_node(state: State) -> dict:
    document_type = state["document_type"]
    merged = "\n\n".join(state["merge_result"])
    report = f"Document Type: {document_type}\n\n{merged}\n"
    return {"final_report": report}


# ============================================================
# Build the graph
# ============================================================
graph_builder = StateGraph(State)

graph_builder.add_node("extract_text_node", extract_text_node)
graph_builder.add_node("classifier_node", classifier_node)
graph_builder.add_node("resume_analyzer_node", resume_analyzer_node)
graph_builder.add_node("invoice_analyzer_node", invoice_analyzer_node)
graph_builder.add_node("contract_analyzer_node", contract_analyzer_node)
graph_builder.add_node("general_analyzer_node", general_analyzer_node)
graph_builder.add_node("response_node", response_node)
graph_builder.add_node("summary_node", summary_node)
graph_builder.add_node("risk_node", risk_node)
graph_builder.add_node("keyword_node", keyword_node)
graph_builder.add_node("final_report_node", final_report_node)

# Sequential intake
graph_builder.add_edge(START, "extract_text_node")
graph_builder.add_edge("extract_text_node", "classifier_node")

# Conditional routing
graph_builder.add_conditional_edges(
    "classifier_node",
    route_function,
    {
        "resume_analyzer_node": "resume_analyzer_node",
        "invoice_analyzer_node": "invoice_analyzer_node",
        "contract_analyzer_node": "contract_analyzer_node",
        "general_analyzer_node": "general_analyzer_node",
    },
)

# All analyzers -> response
for branch in (
    "resume_analyzer_node",
    "invoice_analyzer_node",
    "contract_analyzer_node",
    "general_analyzer_node",
):
    graph_builder.add_edge(branch, "response_node")

# response -> parallel summary/risk/keyword
graph_builder.add_edge("response_node", "summary_node")
graph_builder.add_edge("response_node", "risk_node")
graph_builder.add_edge("response_node", "keyword_node")

# parallel -> final report
for branch in ("summary_node", "risk_node", "keyword_node"):
    graph_builder.add_edge(branch, "final_report_node")

graph_builder.add_edge("final_report_node", END)

graph = graph_builder.compile()


# ============================================================
# FastAPI app
# ============================================================
app = FastAPI(
    title="Enterprise Document Intelligence API",
    description="Upload a PDF + ask a question, get back classification, answer, and full report.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----- Response schema -----
class AnalyzeResponse(BaseModel):
    document_type: str
    answer: str
    final_report: str
    model: Optional[str] = "standard"


class HealthResponse(BaseModel):
    status: str
    service: str


# ----- Routes -----
# NOTE: the "/" route is intentionally NOT defined here so StaticFiles can
# serve index.html (landing page) at the root. Health check moved to /health.

@app.get("/health", response_model=HealthResponse)
def health():
    return {"status": "healthy", "service": "Enterprise Document Intelligence"}


def _extract_ai_answer(messages) -> str:
    """Find the last AI message in the graph result."""
    for msg in reversed(messages):
        # LangGraph message objects
        if hasattr(msg, "type") and getattr(msg, "type", "") == "ai":
            return msg.content
        # raw tuple fallback
        if isinstance(msg, tuple) and len(msg) >= 2 and msg[0] == "ai":
            return msg[1]
    return ""


@app.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze(
    file: UploadFile = File(..., description="PDF document to analyze"),
    message: str = Form(..., description="User's question about the document"),
    model: str = Form("standard", description="Model selector: standard | deep | fast"),
):
    # ---- Validate ----
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    if not message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    # ---- Save uploaded PDF to a safe temp path ----
    safe_name = Path(file.filename).name  # strip any path components
    file_path = UPLOAD_DIR / safe_name
    try:
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        size = file_path.stat().st_size
        if size == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        if size > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Max size is {MAX_UPLOAD_BYTES // (1024*1024)} MB.",
            )

        # ---- Run the LangGraph workflow ----
        initial_state = {
            "pdf_path": str(file_path),
            "messages": [("human", message.strip())],
        }
        result = graph.invoke(initial_state)

        answer = _extract_ai_answer(result.get("messages", []))

        return AnalyzeResponse(
            document_type=result.get("document_type", "unknown"),
            answer=answer,
            final_report=result.get("final_report", ""),
            model=model,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")
    finally:
        # ---- Cleanup ----
        try:
            await file.close()
        except Exception:
            pass
        if file_path.exists():
            try:
                file_path.unlink()
            except Exception:
                pass


# ============================================================
# Static files + frontend routes
# Serve the HTML frontend (landing.html + chatbot.html) from the
# same FastAPI app so you only need ONE server.
# ============================================================
FRONTEND_DIR = Path(__file__).resolve().parent  # folder where main.py lives


# Explicit routes — clean URLs that don't need .html in the path
@app.get("/", include_in_schema=False)
async def serve_landing():
    return FileResponse(FRONTEND_DIR / "landing.html")


@app.get("/chatbot", include_in_schema=False)
async def serve_chatbot():
    return FileResponse(FRONTEND_DIR / "chatbot.html")


# Static files mount — also serves the raw .html files directly,
# e.g. /landing.html, /chatbot.html, plus any CSS/JS/images.
# Mounted LAST so the routes above take priority.
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="static")


# ============================================================
# Entry point
# ============================================================
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
