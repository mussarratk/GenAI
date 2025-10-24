# GenAI - Project


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
