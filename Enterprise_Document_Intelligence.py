import os
import operator
from typing import TypedDict, Annotated
from dotenv import load_dotenv

# FOR RAG application libraries:
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser


# For our LangGraph workflow:
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

load_dotenv()

# ------------------------------------------------------------------
# Embedding model + RAG retriever builder
# ------------------------------------------------------------------
# NOTE: param name is `model_name`, not `model`, for HuggingFaceEmbeddings
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def build_retreive(pdf_path: str):
    """Builds a retriever for the given PDF. Called at runtime per-request,
    NOT at module load time, because we don't know the PDF path in advance."""
    loader = PyPDFLoader(pdf_path)
    document = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(document)

    vectorstore = FAISS.from_documents(chunks, embedding_model)
    return vectorstore.as_retriever(search_kwargs={"k": 4})


# NOTE: removed the old module-level retriever creation
# (resume_reitriver = build_retreive() etc.) — this crashed because
# pdf_path wasn't available yet, and it would have pinned all requests
# to a single static file instead of the document the user uploads.

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.4)


# ------------------------------------------------------------------
# State
# ------------------------------------------------------------------
class State(TypedDict):
    pdf_path: str
    extract_text: str
    document_type: str
    merge_result: Annotated[list, operator.add]   # fixed: operator.add, not str
    document_text: str
    messages: Annotated[list, add_messages]
    final_report: str


NO_RETRIEVAL_NEEDED = "NO_RETRIEVAL_NEEDED"  # single constant used everywhere, avoids typos/case mismatch


# ------------------------------------------------------------------
# Sequential: Extract text
# ------------------------------------------------------------------
def Extract_text_Node(state: State) -> dict:
    pdf_path = state["pdf_path"]

    loader = PyPDFLoader(pdf_path)
    document = loader.load()

    full_text = "\n".join([page.page_content for page in document])

    return {"extract_text": full_text}


# ------------------------------------------------------------------
# Conditional: Classify document type
# ------------------------------------------------------------------
def classifier_node(state: State) -> dict:
    last_message = state["messages"][-1]
    extract_text = state["extract_text"]

    classification_prompt = f"""You are a document classification assistant.
Analyze only the contents of the document text below and classify it into exactly one of these categories:
- resume
- invoice
- contract
User context (for reference only, do not classify based on this alone): {last_message.content}
Document text:
{extract_text}
Return only one word: resume, invoice, or contract.
Do not return explanations, punctuation, markdown, or any extra text."""

    response = llm.invoke(classification_prompt)
    category = response.content.strip().lower()

    if category == "resume":
        category = "resume"
    elif category == "invoice":
        category = "invoice"
    elif category == "contract":
        category = "contract"
    else:
        category = "unknown"

    return {"document_type": category}


# ------------------------------------------------------------------
# Conditional routing targets: type-specific analyzers
# ------------------------------------------------------------------
def Resume_analyzer_Node(state: State) -> dict:
    query = state["messages"][-1].content
    retriever = build_retreive(state["pdf_path"])
    document = retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in document])
    return {"document_text": context}


def Invoice_analyzer_Node(state: State) -> dict:
    query = state["messages"][-1].content   # fixed: was missing .content
    retriever = build_retreive(state["pdf_path"])
    document = retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in document])
    return {"document_text": context}


def Contract_analyzer_Node(state: State) -> dict:
    query = state["messages"][-1].content   # fixed: was missing .content
    retriever = build_retreive(state["pdf_path"])
    document = retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in document])
    return {"document_text": context}


def general_analyzer_Node(state: State) -> dict:
    """If the document type couldn't be classified into resume/invoice/contract."""
    return {"document_text": NO_RETRIEVAL_NEEDED}   # fixed: was a set, not a dict


def Route_function(state: State):
    # fixed: compare against lowercase values (classifier_node always returns lowercase)
    if state["document_type"] == "resume":
        return "Resume_analyzer_Node"
    elif state["document_type"] == "invoice":
        return "Invoice_analyzer_Node"
    elif state["document_type"] == "contract":       # fixed: was checking document_text, not document_type
        return "Contract_analyzer_Node"
    else:
        return "general_analyzer_Node"


# ------------------------------------------------------------------
# Response generation (answers the user's question about the document)
# ------------------------------------------------------------------
def response_node(state: State) -> dict:
    document_type = state["document_type"]
    context = state["document_text"]
    query = state["messages"][-1].content

    if context == NO_RETRIEVAL_NEEDED:
        prompt = f"""You are an expert assistant for {document_type} documents.

The detected document type is: {document_type}

Answer the user's question using your own knowledge. Tailor your answer according to the document type:
- resume → answer as a resume expert
- invoice → answer as an invoice/accounting expert
- contract → answer as a legal contract expert

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
- resume → career/resume assistant
- invoice → financial/invoice assistant
- contract → legal contract assistant"""

    result = llm.invoke(prompt)
    response = result.content.strip()

    # fixed: `response` is already a plain string, calling .content on it would crash
    return {"messages": [("ai", response)]}


# ------------------------------------------------------------------
# Parallel workflow: summary / risk / keywords
# ------------------------------------------------------------------
def summary_Node(state: State) -> dict:
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
    summary = result.content.strip()

    return {"merge_result": [f"Summary:\n{summary}"]}


def Risk_Node(state: State) -> dict:
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
    risks = result.content.strip()

    return {"merge_result": [f"Risks:\n{risks}"]}


def Keyword_Node(state: State) -> dict:
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
    keywords = result.content.strip()

    return {"merge_result": [f"Keywords:\n{keywords}"]}


# ------------------------------------------------------------------
# Sequential (final): merge parallel outputs into one report
# ------------------------------------------------------------------
def final_report_Node(state: State) -> dict:
    document_type = state["document_type"]
    merged = "\n\n".join(state["merge_result"])

    report = f"""Document Type: {document_type}

{merged}
"""
    return {"final_report": report}


# ------------------------------------------------------------------
# Build the graph
# ------------------------------------------------------------------
graph_builder = StateGraph(State)

graph_builder.add_node("Extract_text_Node", Extract_text_Node)
graph_builder.add_node("classifier_node", classifier_node)
graph_builder.add_node("Resume_analyzer_Node", Resume_analyzer_Node)
graph_builder.add_node("Invoice_analyzer_Node", Invoice_analyzer_Node)
graph_builder.add_node("Contract_analyzer_Node", Contract_analyzer_Node)
graph_builder.add_node("general_analyzer_Node", general_analyzer_Node)
graph_builder.add_node("response_node", response_node)
graph_builder.add_node("summary_Node", summary_Node)
graph_builder.add_node("Risk_Node", Risk_Node)
graph_builder.add_node("Keyword_Node", Keyword_Node)
graph_builder.add_node("final_report_Node", final_report_Node)

# Sequential intake
graph_builder.add_edge(START, "Extract_text_Node")
graph_builder.add_edge("Extract_text_Node", "classifier_node")

# Conditional routing
graph_builder.add_conditional_edges(
    "classifier_node",
    Route_function,
    {
        "Resume_analyzer_Node": "Resume_analyzer_Node",
        "Invoice_analyzer_Node": "Invoice_analyzer_Node",
        "Contract_analyzer_Node": "Contract_analyzer_Node",
        "general_analyzer_Node": "general_analyzer_Node",
    },
)

# All analyzer branches converge into response_node
graph_builder.add_edge("Resume_analyzer_Node", "response_node")
graph_builder.add_edge("Invoice_analyzer_Node", "response_node")
graph_builder.add_edge("Contract_analyzer_Node", "response_node")
graph_builder.add_edge("general_analyzer_Node", "response_node")

# Parallel workflow: response_node fans out to all three at once
graph_builder.add_edge("response_node", "summary_Node")
graph_builder.add_edge("response_node", "Risk_Node")
graph_builder.add_edge("response_node", "Keyword_Node")

# All three converge into the final report
graph_builder.add_edge("summary_Node", "final_report_Node")
graph_builder.add_edge("Risk_Node", "final_report_Node")
graph_builder.add_edge("Keyword_Node", "final_report_Node")

graph_builder.add_edge("final_report_Node", END)

graph = graph_builder.compile()


# ------------------------------------------------------------------
# User input / entry point
# ------------------------------------------------------------------
def run():
    pdf_path = input("Enter the path to your PDF: ").strip()

    if not os.path.exists(pdf_path):
        print(f"File not found: {pdf_path}")
        return

    user_query = input("What would you like to know about this document? ").strip()

    initial_state = {
        "pdf_path": pdf_path,
        "messages": [("human", user_query)],
    }

    result = graph.invoke(initial_state)

    print("\n=== Document Type ===")
    print(result["document_type"])

    print("\n=== Answer ===")
    print(result["messages"][-1].content)

    print("\n=== Final Report ===")
    print(result["final_report"])


if __name__ == "__main__":
    run()