# 1Q)For Question 10 (return policy): did your system hallucinate? If yes, what exact change to the system prompt fixed it? Paste both versions
The system did not hallucinate when answering the question.

Instead of generating a fabricated return policy, the model analyzed the retrieved customer reviews and summarized the issues customers faced while attempting returns.

The response was grounded in the retrieved review context produced by the RAG pipeline.

The LLM used in the system, Mistral, did not invent any policy details such as a specific number of return days or refund guarantees.


#  2 Compare your custom RAG answer quality vs LangChain RAG on 3 questions. Are they the same? If different, why?


The system was tested using two implementations:

1. A **custom Retrieval-Augmented Generation (RAG) pipeline** built manually using vector search with FAISS.
2. A framework-based RAG pipeline implemented using LangChain.

Both systems used the same embeddings and the same LLM model (Mistral) to ensure a fair comparison.

---

# Questions Used for Comparison

The following queries were used to evaluate both pipelines:

1. What are the most common complaints about kitchen appliances?
2. Are there delivery delays mentioned in reviews?
3. What do customers say about DreamRest mattress?

---

# Observation of Results

### Example Output (Custom RAG)

The custom RAG pipeline produced answers such as:

- Kitchen appliance complaints included **noise issues, quality concerns, and safety hazards**.
- Delivery delays were mentioned, including **10-day replacement delays and 3-week delivery times**.
- Customer feedback for DreamRest mattresses showed **mixed opinions with both positive and average reviews**.

The answers were **factually grounded in the retrieved reviews** and clearly summarized key points from the dataset.

---

# Comparison with LangChain RAG

| Aspect | Custom RAG | LangChain RAG |
|------|------|------|
| Answer accuracy | Accurate and grounded | Similar accuracy |
| Response structure | Simple summary | Slightly more structured |
| Latency | ~1.4 – 2.2 seconds | Similar latency |
| Implementation complexity | Requires manual pipeline setup | Easier to implement |

---

# Key Differences

## 1. Prompt and Document Formatting

LangChain automatically formats retrieved documents before sending them to the LLM, which can improve the readability and structure of the generated answers.

In the custom RAG implementation, document formatting and prompt construction must be handled manually.

---

## 2. Pipeline Control

The custom implementation provides **greater control over the retrieval pipeline**, including:

- Embedding generation
- Similarity search
- Prompt construction

LangChain abstracts many of these steps into predefined chains.

---

## 3. System Latency

The custom RAG pipeline produced responses in approximately **1.4 to 2.2 seconds**, which is comparable to the LangChain implementation since both rely on the same LLM and retrieval system.

---

# Final Conclusion

The outputs from the custom RAG and LangChain RAG pipelines were **largely similar in terms of answer accuracy and relevance**.

The main difference lies in **implementation convenience**:

- Custom RAG provides **greater flexibility and control** over the pipeline.
- LangChain simplifies development through built-in chains and document handling.

Therefore, while both approaches produce comparable results, LangChain is generally **easier to implement**, whereas custom RAG offers **more control over system behavior**.



# LangChain Chain Types: Stuff, Map Reduce, and Refine

In LangChain, different **chain types** are used to combine multiple retrieved documents before sending them to the Large Language Model (LLM) for generating an answer.

The most common chain types are **stuff**, **map_reduce**, and **refine**.

---

# 1. Stuff Chain

The **stuff chain** is the simplest document combination strategy.

In this approach, all retrieved documents are **combined together ("stuffed") into a single prompt** and sent to the LLM at once.

### Workflow

Retrieve documents → Combine all documents into one prompt → Send to LLM → Generate answer

### Advantages

- Simple and fast
- Works well for small document sets
- Low processing overhead

### Limitations

- If many documents are retrieved, the prompt may exceed the **token limit**
- The model may ignore some information if the context is too large

### When to Use

Use the **stuff chain** when:

- The number of retrieved documents is small
- Documents are short
- Token limits are not a concern

This is commonly used in basic **RAG pipelines**.

---

# 2. Map Reduce Chain

The **map_reduce chain** processes each document **individually first**, then combines the results.

### Workflow

1. **Map Step**  
   The LLM generates a partial answer for each document separately.

2. **Reduce Step**  
   The partial answers are combined into a final summarized answer.

### Advantages

- Handles **large document collections**
- Reduces risk of exceeding token limits
- Allows the model to analyze documents independently

### Limitations

- Slower because multiple LLM calls are required
- More expensive in terms of token usage

### When to Use

Use **map_reduce** when:

- The dataset contains **many documents**
- Documents are long
- You need scalable document processing

---

# 3. Refine Chain

The **refine chain** builds the answer **incrementally** by refining it as new documents are processed.

### Workflow

1. Generate an initial answer using the **first document**
2. Pass the answer and the **next document** to the LLM
3. The LLM **refines the answer**
4. Repeat until all documents are processed

### Advantages

- Allows answers to **gradually improve**
- Works well when documents add **incremental information**

### Limitations

- Slower because documents are processed sequentially
- If the first answer is weak, later refinements may still be biased

### When to Use

Use **refine** when:

- Information is spread across documents
- Each document adds additional context to the answer

---

# Summary Comparison

| Chain Type | How It Works | Best Use Case |
|---|---|---|
| Stuff | All documents are inserted into one prompt | Small document sets |
| Map Reduce | Each document processed separately then combined | Large document collections |
| Refine | Answer refined progressively with each document | Incremental knowledge across documents |

---

# Conclusion

The **stuff chain** is the simplest and fastest method but works best with small datasets.  
**Map_reduce** is more scalable for large document collections, while **refine** is useful when answers need to be progressively improved using multiple sources.
















# 4)Effect of Changing `top_k` in RAG Retrieval

In a RAG system, **top_k** controls how many documents (reviews) are retrieved from the vector database for answering a query.

### top_k = 1
- Retrieves only the **most similar review**
- **Pros:** Very fast, low token cost  
- **Cons:** May miss important information from other reviews

### top_k = 5 (Current Setting)
- Retrieves **five relevant reviews**
- **Pros:** Balanced context and better answer quality  
- **Cons:** Moderate token usage

### top_k = 15
- Retrieves **many reviews**
- **Pros:** Richer context and more coverage  
- **Cons:** Higher token cost, slower responses, possible irrelevant information

### Trade-off

| top_k | Context Quality | Token Cost |
|------|------|------|
| 1 | Low | Very Low |
| 5 | Balanced | Moderate |
| 15 | High | High |

**Conclusion:**  
`top_k = 5` provides a good balance between **context richness and token efficiency**.