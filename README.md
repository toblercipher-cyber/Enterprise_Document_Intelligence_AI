<img width="1360" height="648" alt="Enterprise_document_intelligence-1" src="https://github.com/user-attachments/assets/57835211-dda4-4696-a1e4-6558f2b69de5" />
<img width="1360" height="646" alt="Enterprise_document_intelligence-2" src="https://github.com/user-attachments/assets/1aebaf10-ca0d-4bdd-b595-9f72b3c7eba6" />
<img width="1360" height="643" alt="Enterprise_document_intelligence-3" src="https://github.com/user-attachments/assets/a2d5fbe2-0a31-4e14-a342-2559bc7905ae" />
<img width="1360" height="641" alt="Enterprise_document_intelligence-4" src="https://github.com/user-attachments/assets/40b2d708-e0f4-46b2-908a-1ada8d9b30fb" />


# 📄 Enterprise Document Intelligence using LangGraph

An AI-powered Enterprise Document Intelligence System built with **LangGraph** that demonstrates **Sequential**, **Conditional**, and **Parallel Workflows** in a real-world document analysis application.

The system automatically identifies the uploaded document type (Resume, Invoice, or Employment Contract), performs document-specific analysis, and generates a comprehensive AI-powered report.

---

## 🚀 Features

- 📂 Upload PDF documents
- 📝 Extract text from PDF files
- 🧠 Automatically detect document type
- 🔀 Dynamic Conditional Routing using LangGraph
- ⚡ Parallel AI analysis for faster processing
- 📋 Generate professional document reports
- 🔍 Extract important keywords
- ⚠️ Identify potential risks and issues
- 📖 Generate concise document summaries
- 🛠️ Modular LangGraph architecture

---

## 🏗️ Workflow Architecture

```text
Upload PDF
      │
      ▼
Extract Text
      │
      ▼
Detect Document Type
      │
      ▼
──────── Conditional Workflow ────────
│              │               │
Resume      Invoice      Contract
Analyzer     Analyzer      Analyzer
│              │               │
──────────────┬────────────────┘
              │
              ▼
──────── Parallel Workflow ────────
│             │                 │
Summary     Risks          Keywords
│             │                 │
─────────────┴──────────────────┘
              │
              ▼
Generate Final Report
```

---

## 🧩 LangGraph Concepts Used

- ✅ Sequential Workflow
- ✅ Conditional Workflow
- ✅ Parallel Workflow
- ✅ State Management
- ✅ Conditional Routing
- ✅ Reducers (State Merging)
- ✅ Prompt Engineering
- ✅ LLM Orchestration
- ✅ Exception Handling

---

## 📂 Supported Document Types

- 📄 Resume
- 🧾 Invoice
- 📜 Employment Contract

---

## 🛠️ Tech Stack

- Python
- LangGraph
- LangChain
- Groq LLM (Llama 3.3 70B)
- PyPDF / PDF Text Extraction
- Python Typing (TypedDict & Annotated)

---

## 🎯 Project Objective

The goal of this project is to demonstrate how multiple LangGraph workflows can be combined to build an enterprise-level AI application.

The system first extracts text from a PDF, identifies the document type through a **Conditional Workflow**, routes it to the appropriate analyzer, performs **Parallel Analysis** (Summary, Risk Detection, and Keyword Extraction), and finally generates a structured AI-powered report.

---

## 📸 Example Output

```text
Document Type:
Employment Contract

Summary:
...

Risks:
...

Keywords:
...

Recommendations:
...
```

---

## 📚 Learning Outcomes

This project helped reinforce the following LangGraph concepts:

- Building stateful AI workflows
- Dynamic workflow routing
- Parallel node execution
- State reducers
- Prompt engineering
- Modular workflow design
- Enterprise AI architecture

---

## 👨‍💻 Author

**Syed Abdul Rehman**

AI Engineer | LangChain & LangGraph Developer | Building Enterprise AI Applications

---
⭐ If you found this project useful, consider giving it a star!
