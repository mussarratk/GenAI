# GenAI - Project
---
# 📄 PDF RAG on Databricks — Document Q&A with LangChain + Vector Search

**End-to-end Retrieval-Augmented Generation pipeline: PDF extraction → chunking → Delta storage → Vector Search → LLM-generated answers, built entirely on the Databricks Lakehouse Platform.**

![Databricks](https://img.shields.io/badge/Databricks-FF3621?style=flat&logo=databricks&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=flat&logo=langchain&logoColor=white)
![Delta Lake](https://img.shields.io/badge/Delta%20Lake-00ADD8?style=flat)
![Vector Search](https://img.shields.io/badge/Vector%20Search-RAG-green)
![Llama 3.3](https://img.shields.io/badge/Llama%203.3%2070B-Model%20Serving-purple)

---

## 1. Project Overview

This project implements a **classic document-grounded RAG pipeline** on Databricks: a PDF is parsed into raw text, split into overlapping chunks, persisted as a Delta table, embedded into a Databricks Vector Search index, and retrieved at query time to ground a Llama 3.3 70B model's answer — orchestrated with **LangChain Expression Language (LCEL)**. Unlike a keyword search over the document, the pipeline lets a user ask a natural-language question (e.g., *"Who is a Data Fiduciary?"*) and get an answer grounded in the actual source PDF, with the LLM instructed not to use irrelevant retrieved context.

## 2. Business Problem

Dense compliance/legal PDFs (this project uses a data-protection-act style document) are long, technical, and slow to search manually — finding a single definition (e.g., "Data Fiduciary") means reading through pages of legal text. Static keyword search (Ctrl+F) fails when the user's phrasing doesn't match the document's exact wording, and copy-pasting sections into a general-purpose chatbot risks answers that aren't actually grounded in the source document.

## 3. Solution

Build a **RAG pipeline that only answers from the document itself**:
1. Extract the PDF's full text programmatically (no manual copy-paste).
2. Split it into retrieval-sized chunks with controlled overlap so context isn't cut mid-thought.
3. Persist chunks in a governed Delta table, then index them in Databricks Vector Search for semantic retrieval.
4. At query time, retrieve the top-k most relevant chunks and pass them into a constrained prompt template that explicitly tells the LLM to ignore irrelevant context — reducing hallucination risk.
5. Generate the final answer with a Databricks-hosted foundation model (Llama 3.3 70B) via Model Serving, wired together as a composable LangChain chain.

## 4. Architecture Design

```
        ┌───────────────────────────┐
        │   Source PDF (Volume)     │
        │   /Volumes/.../dpact.pdf  │
        └─────────────┬─────────────┘
                       │  PyMuPDF (fitz) — page-by-page text extraction
                       ▼
        ┌───────────────────────────┐
        │      Raw extracted text    │
        └─────────────┬─────────────┘
                       │  LangChain RecursiveCharacterTextSplitter
                       │  chunk_size=500, chunk_overlap=100
                       ▼
        ┌───────────────────────────┐
        │   Chunked documents (docs) │
        │   id_pk + page_content     │
        └─────────────┬─────────────┘
                       │  spark.createDataFrame(...).write.format("delta")
                       ▼
        ┌───────────────────────────┐
        │  Delta Table               │
        │  workspace.default.my_data_chunks
        └─────────────┬─────────────┘
                       │  Databricks Vector Search (embed + index)
                       ▼
        ┌───────────────────────────┐
        │  Vector Search Index        │
        │  workspace.default.my_index │
        └─────────────┬─────────────┘
                       │  similarity_search() /
                       │  DatabricksVectorSearch.as_retriever(k=2)
                       ▼
        ┌───────────────────────────┐
        │   Retrieved chunks          │
        │   → format_context()        │
        └─────────────┬─────────────┘
                       │  ChatPromptTemplate (system: grounded-answer
                       │  instructions + context | user: question)
                       ▼
        ┌───────────────────────────┐
        │  ChatDatabricks LLM         │
        │  databricks-meta-llama-3-3-70b-instruct
        │  (Model Serving endpoint)   │
        └─────────────┬─────────────┘
                       │  LCEL chain: prompt | model | StrOutputParser
                       ▼
        ┌───────────────────────────┐
        │   Grounded natural-language │
        │   answer                    │
        └───────────────────────────┘

        (Also callable as a deployed REST endpoint via requests.post
         once the chain is registered/served — seen at the end of
         the notebook.)
```

## 5. Tech Stack

| Layer | Technology |
|---|---|
| Platform | Databricks (Notebooks, Unity Catalog Volumes) |
| PDF extraction | PyMuPDF (`fitz`) |
| Chunking | LangChain `RecursiveCharacterTextSplitter` |
| Storage | Delta Lake (chunk table with primary key `id_pk`) |
| Vector store | Databricks Vector Search (`VectorSearchClient`, `similarity_search`) |
| RAG orchestration | LangChain / `databricks-langchain` (LCEL: prompt → model → parser) |
| Retriever integration | `DatabricksVectorSearch` LangChain retriever (`as_retriever(k=2)`) |
| LLM | Llama 3.3 70B Instruct via Databricks Model Serving (`ChatDatabricks`) |
| Prompt engineering | `ChatPromptTemplate` (system + user message separation) |
| Serving | Deployed REST endpoint invocation (`requests.post`) for production inference |
| Language | Python |

## 6. Implementation

1. **Environment setup** — installed `databricks-sdk`, `databricks-langchain`, `databricks-agents`, `mlflow[databricks]`, `databricks-vectorsearch`, `langchain`, `bs4`, `markdownify`, `PyMuPDF` and restarted the Python environment — the standard Databricks GenAI dependency stack.
2. **LLM sanity check** — instantiated `ChatDatabricks` against the `databricks-meta-llama-3-3-70b-instruct` serving endpoint and validated it with a direct `.invoke()` call before building any retrieval logic around it.
3. **PDF extraction** — read the source PDF from a Unity Catalog Volume and extracted full text page-by-page with PyMuPDF (`fitz`).
4. **Chunking** — split the raw text using `RecursiveCharacterTextSplitter` (500-char chunks, 100-char overlap) to balance retrieval precision against context completeness, then converted chunks to a Pandas/Spark DataFrame with a generated `id_pk` primary key.
5. **Persisting chunks** — wrote the chunked `(id_pk, page_content)` table to Delta (`workspace.default.my_data_chunks`) as the durable, queryable source of truth for the index.
6. **Vector Search** — indexed the chunk table in Databricks Vector Search and validated retrieval with a raw `similarity_search()` call before wrapping it in LangChain's `DatabricksVectorSearch` retriever abstraction (`as_retriever(k=2)`) for chain composition.
7. **Context formatting** — wrote a `format_context()` helper to flatten retrieved LangChain `Document` objects into a single "Passage: ..." string suitable for prompt injection.
8. **Prompt template & config** — externalized the model endpoint, vector index name, and system prompt into a `chain_config` dict, then built a `ChatPromptTemplate` with a system message (grounded-answer instructions + `{context}`) and a user message (`{question}`) — separating configuration from chain logic.
9. **Chain composition (LCEL)** — composed the final RAG chain as `prompt | model | StrOutputParser()`, LangChain's declarative pipe syntax, and invoked it end-to-end with a real question and retrieved context.
10. **Production path** — the notebook closes with a `requests.post` call to a deployed endpoint URL, indicating the chain is intended to be registered/served (e.g., via MLflow + Model Serving) and called over REST for real-time inference in an application.

## 7. Code Example

**Chunking + indexing (ingestion path):**
```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
docs = text_splitter.create_documents([raw_text])

pd_docs = pd.DataFrame([doc.dict() for doc in docs])
pd_docs.insert(0, "id_pk", range(1, len(pd_docs) + 1))

spark_df = spark.createDataFrame(pd_docs[['id_pk', 'page_content']])
spark_df.write.format("delta").mode("overwrite").saveAsTable("workspace.default.my_data_chunks")
```

**Retrieval-augmented chain (query path):**
```python
from databricks_langchain import DatabricksVectorSearch, ChatDatabricks
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

vector_store = DatabricksVectorSearch(index_name="workspace.default.my_index")
retriever = vector_store.as_retriever(search_kwargs={"k": 2})

prompt = ChatPromptTemplate.from_messages([
    ("system", chain_config["llm_prompt_template"]),
    ("user", "{question}")
])

chain = prompt | ChatDatabricks(endpoint=chain_config["llm_model_serving_endpoint_name"]) | StrOutputParser()
answer = chain.invoke({"question": "Who is a Data Fiduciary?", "context": format_context(retriever.invoke("Who is a Data Fiduciary?"))})
```

## 8. Key Highlights

- Full **ingestion-to-inference RAG loop** built from a raw PDF, not a pre-cleaned dataset — extraction, chunking, and indexing all handled explicitly.
- Used **both** the raw Databricks Vector Search client (`VectorSearchClient.similarity_search`) **and** the LangChain retriever abstraction (`DatabricksVectorSearch.as_retriever`) — showing familiarity with the platform SDK and the higher-level orchestration framework.
- **LCEL (LangChain Expression Language)** used for chain composition (`prompt | model | parser`) — the modern, declarative LangChain pattern over legacy chain classes.
- Prompt explicitly instructs the model to **ignore irrelevant retrieved context**, a practical hallucination-mitigation technique rather than trusting the LLM's default behavior.
- Chain configuration (endpoint names, index name, prompt template) externalized into a `chain_config` dict — a step toward parameterized, environment-portable RAG pipelines.
- Notebook ends with a **deployed REST endpoint call**, showing the intended path to production serving rather than stopping at notebook-only experimentation.

## 9. Key Concepts

- **RAG (Retrieval-Augmented Generation)** — grounding LLM output in retrieved source content instead of relying purely on parametric knowledge.
- **Chunking strategy** — chunk size vs. overlap trade-off (500/100 here) and its effect on retrieval precision vs. context continuity.
- **Vector Search / semantic retrieval** — embedding-based similarity search vs. keyword search.
- **LCEL (LangChain Expression Language)** — composing chains declaratively with the `|` pipe operator.
- **Prompt engineering** — system vs. user message separation, explicit grounding/anti-hallucination instructions.
- **Foundation Model Serving** — calling a hosted LLM (Llama 3.3 70B) via a managed Databricks endpoint instead of self-hosting.

## 10. Skills Gained

- PDF text extraction with PyMuPDF.
- Designing and tuning a chunking strategy for retrieval quality.
- Persisting unstructured-derived data into governed Delta tables.
- Building and querying a Databricks Vector Search index, both via SDK and LangChain integration.
- Composing RAG chains with LangChain Expression Language (LCEL).
- Prompt template design for grounded, hallucination-resistant answers.
- Integrating Databricks Foundation Model Serving (`ChatDatabricks`) into an application chain.
- Understanding the path from notebook experimentation to a deployed, REST-callable inference endpoint.

## 11. Business Outcome

- Converts a static, hard-to-search compliance/legal PDF into an **interactive natural-language Q&A interface**, cutting manual document lookup time.
- Reduces hallucination risk versus a bare LLM prompt by grounding every answer in retrieved source passages and instructing the model to disregard irrelevant context.
- Establishes a **reusable RAG pattern** (extract → chunk → index → retrieve → generate) that can be pointed at any new PDF/document set with minimal rework.
- Demonstrates a clear path to production — the chain is structured to be served behind a REST endpoint, not just run ad hoc in a notebook.

## 12. Repo Structure (suggested)

```
pdf-rag-databricks/
├── README.md
├── notebooks/
│   └── endtoend.ipynb          # PDF extraction → chunking → indexing → RAG chain
├── data/
│   └── (sample PDF placeholder — do not commit real source docs)
└── docs/
    └── architecture.png
```

## 13. How to Explain This in an Interview (cheat sheet)

- **"Walk me through the project"** → "I built a document-grounded RAG pipeline on Databricks — extract text from a PDF with PyMuPDF, chunk it with LangChain's recursive splitter, persist chunks to Delta, index them in Databricks Vector Search, then retrieve and answer with a Llama 3.3 70B model wired together as an LCEL chain."
- **"Why chunk at 500 characters with 100 overlap?"** → "It's a balance — small enough for precise retrieval and to stay within embedding/context limits, with overlap so a sentence or definition split across a chunk boundary isn't lost entirely from either chunk."
- **"Why use both the raw Vector Search client and the LangChain retriever?"** → "I validated retrieval quality directly against the platform SDK first, then wrapped it in LangChain's retriever interface so it composes cleanly with the rest of the chain — validate low-level, integrate high-level."
- **"How do you reduce hallucination here?"** → "Two ways: retrieval grounds the answer in real source text, and the system prompt explicitly tells the model to ignore irrelevant retrieved passages rather than forcing an answer from them."
- **"What would you productionize further?"** → "Register the chain with MLflow, evaluate it (e.g., LLM-as-a-Judge or retrieval recall@k), and deploy it behind Model Serving — the notebook's closing REST call shows that's the intended direction, just not fully wired up yet."

> **Note on this pass:** a few notebook cells reference variables not defined earlier in the shown code (`relevant_docs`, `StrOutputParser`, `requests`/`ENDPOINT_URL`/`headers` in the final cell) — typical of iterative notebook development. Worth quickly cleaning those imports/variable names before treating this as a polished portfolio piece, since an interviewer skimming the raw notebook (not just this README) could spot it.


---
<details>


## 🌟 **Generative AI Project: Enterprise RAG and Summarization on Databricks**

---

### **Project Summary & Key Contributions**

* **Goal:** Designed and implemented a full-stack Generative AI solution on the **Databricks Data Intelligence Platform** (DIP), featuring a **Review Summarizer Model** and an **Enterprise-grade Retrieval-Augmented Generation (RAG) system** for secure, context-aware Q\&A.
* **Technologies:** Databricks (Workspaces, Unity Catalog, Vector Search, Model Serving, MLflow), LLMs (e.g., `databricks-meta-llama-3-70b-instruct`), LangChain, PyMuPDF, Delta Lake.

---

### **Project Details**

1.  **High-Performance Summarization Model Development**
    * **Access & Engineering:** Accessed and utilized Databricks-hosted **LLM Foundational Models** programmatically and via the **LLM Playground**.
    * **Prompt Engineering:** Mastered **Prompt Engineering** techniques (e.g., Use of Delimiters, Structured Output, multi-task prompts) within Databricks Notebooks to optimize summarizer performance for product reviews (including sentiment and topic inference).
    * **Batch & Real-Time Deployment:**
        * **Batch:** Deployed the summarization model (`t5-small` or similar) using **Spark UDFs** and **Databricks `ai_query` function** for high-throughput batch inference over a Delta table of product reviews.
        * **Real-Time:** Implemented **real-time model serving** by creating and invoking a **Databricks Model Serving Endpoint** using Python `requests`.

2.  **Enterprise RAG System with Databricks Vector Search**
    * **Data Ingestion & Transformation:** Utilized **LangChain** and **PyMuPDF** to extract content from PDF documents, applied `langchain.text_splitter` for optimal **chunking**, and persisted the chunks into a **Delta Lake table** within **Unity Catalog (UC)**.
    * **Vectorization & Indexing:** Created an efficient **Vector Search Endpoint** and indexed the chunked data, enabling fast and accurate **similarity search** (e.g., retrieving the Top-K relevant documents).
    * **RAG Chain Construction:** Built a robust **LangChain pipeline** using **`ChatDatabricks`** for LLM interaction, a **retriever** configured with **Vector Search**, a **context formation step**, and a refined **prompt template** to construct the final RAG chain for answering domain-specific queries.

3.  **MLOps and Project Lifecycle Management**
    * **MLflow Integration:** Managed the entire GenAI project lifecycle using **MLflow**, including **experiment tracking** (logging artifacts, model metrics), **model versioning**, and **registration** into **Unity Catalog** with aliases.
    * **Evaluation:** Conducted comprehensive **LLM Model Evaluation** using standard metrics, component-wise analysis, and advanced techniques like **'LLM-as-a-Judge'** and **RAG-specific evaluation** to ensure response quality and grounding.

---



  
</details>


<img width="1098" height="531" alt="image" src="https://github.com/user-attachments/assets/c05373f2-dcf3-4eac-824c-8148bf86c93c" />

<img width="1345" height="589" alt="image" src="https://github.com/user-attachments/assets/5402c6d6-ffaa-4bcd-9e51-c86d3434684a" />


<img width="1358" height="609" alt="image" src="https://github.com/user-attachments/assets/5c4af85f-a301-446f-b3d0-6e650ae50d17" />

<img width="1355" height="397" alt="image" src="https://github.com/user-attachments/assets/969326bb-748e-4a94-8421-c43ecd6fa5da" />

<img width="1355" height="635" alt="image" src="https://github.com/user-attachments/assets/e5177d4c-6e60-4a55-b545-d8c248826555" />

<img width="1346" height="622" alt="image" src="https://github.com/user-attachments/assets/d8fced8d-f628-4b23-9073-db4338a1a117" />

<img width="1353" height="348" alt="image" src="https://github.com/user-attachments/assets/50323a70-f34a-41e4-af0a-b3aafdc7f806" />

<img width="1350" height="568" alt="image" src="https://github.com/user-attachments/assets/138e35d8-0531-4193-8b06-b3941276c37d" />

<img width="1351" height="605" alt="image" src="https://github.com/user-attachments/assets/f841ca23-307c-4ecd-8f3c-9debd440b31f" />

<img width="1354" height="518" alt="image" src="https://github.com/user-attachments/assets/680d8412-811a-43ef-a448-abdd00641331" />

<img width="1341" height="440" alt="image" src="https://github.com/user-attachments/assets/f285fb01-0558-4e07-a8bd-eaca241dde61" />

<img width="1366" height="630" alt="image" src="https://github.com/user-attachments/assets/be8844ef-999f-4057-963a-9a1e5ded6c46" />

<img width="1309" height="632" alt="image" src="https://github.com/user-attachments/assets/0359e4e0-2baf-44e8-8fff-9d72080d901b" />

<img width="1165" height="457" alt="image" src="https://github.com/user-attachments/assets/941c0781-3971-4bfe-8067-60624f4a1421" />


# Real Time Deployment of My Model (Summarizer) - Model Serving -[8. Access Models using Databricks, 73.]

<img width="1353" height="521" alt="image" src="https://github.com/user-attachments/assets/b84a063e-ca74-4c48-a3e9-1c86940df366" />

<img width="1356" height="612" alt="image" src="https://github.com/user-attachments/assets/3b13231a-81f3-4ab7-9c9e-06c6828cd920" />

<img width="1346" height="594" alt="image" src="https://github.com/user-attachments/assets/bce48da8-8f22-472b-bd1f-77b0ca3fdbda" />

<img width="1344" height="620" alt="image" src="https://github.com/user-attachments/assets/179536a1-0093-4129-a2f5-67a53b24e75d" />

<img width="1359" height="633" alt="image" src="https://github.com/user-attachments/assets/f47279b2-c1e9-449e-9a6e-5a552f658c72" />


-----

<img width="1122" height="564" alt="image" src="https://github.com/user-attachments/assets/1137cc93-38c7-4eb3-8e64-d93c02da7d60" />



<img width="1088" height="627" alt="image" src="https://github.com/user-attachments/assets/64cbd12e-1f6d-4c44-8698-db1a6278e776" />
<img width="1355" height="620" alt="image" src="https://github.com/user-attachments/assets/cbe2926c-018e-4a19-8fe9-ad18e4fab754" />
<img width="1354" height="626" alt="image" src="https://github.com/user-attachments/assets/adeba934-4172-482e-8d9e-725500fd26cb" />
<img width="759" height="637" alt="image" src="https://github.com/user-attachments/assets/e2a761fd-ac53-4d3f-b48f-3fa0708c898a" />
<img width="1353" height="620" alt="image" src="https://github.com/user-attachments/assets/bca48910-c684-4a48-b4ce-e269ee6262c3" />
<img width="1353" height="622" alt="image" src="https://github.com/user-attachments/assets/b08ff37b-1387-4c12-af24-3969c2c22ea5" />


----
