# Basic RAG System with Embeddings

A simple Retrieval-Augmented Generation implementation using sentence embeddings and cosine similarity for semantic document retrieval.

## Overview

This notebook demonstrates core RAG concepts:
1. Document embedding generation
2. Semantic similarity computation
3. Context retrieval based on queries
4. Answer generation with retrieved context

## Methodology

### Components

**Document Database**
- 15 machine learning concept documents
- Structured storage with embeddings

**Embedding Model**
- Model: `all-MiniLM-L6-v2` (SentenceTransformer)
- Dimension: 384
- Fast and efficient for semantic search

**Retrieval Mechanism**
- Cosine similarity between query and document embeddings
- Top-k document selection
- Ranked by relevance score

### Workflow

1. **Document Preparation**
   - Load knowledge base
   - Create structured database

2. **Embedding Generation**
   - Convert documents to dense vectors
   - Store embeddings for retrieval

3. **Query Processing**
   - Encode user query
   - Compute similarity with all documents
   - Retrieve top-k matches

4. **Context Assembly**
   - Combine retrieved documents
   - Provide context for answer generation

## Key Features

- **Semantic Search**: Beyond keyword matching
- **Cosine Similarity**: Efficient similarity metric
- **Top-K Retrieval**: Configurable result count
- **Visualization**: Similarity score analysis
- **Evaluation**: Quality metrics and statistics

## Requirements

```bash
pip install sentence-transformers numpy scikit-learn pandas matplotlib
```

## Usage

### Basic Retrieval

```python
query = "How do neural networks work?"
results = retrieve_documents(query, top_k=3)
```

### Answer Generation

```python
answer_data = generate_answer(query, top_k=3)
print(answer_data['context'])
```

### Interactive Queries

```python
interactive_query("What is supervised learning?", top_k=3)
```

## Implementation Details

**Embedding Model**
- Pre-trained on large text corpus
- Captures semantic meaning
- Efficient inference

**Similarity Computation**
```
similarity = (query_vector · doc_vector) / (||query_vector|| × ||doc_vector||)
```

**Ranking**
- Documents sorted by similarity score
- Higher scores indicate better matches
- Threshold filtering optional

## Results

The notebook demonstrates:
- Effective semantic retrieval
- Query-document matching
- Similarity score distributions
- Retrieval quality metrics

## Evaluation Metrics

- **Max Similarity**: Best match score
- **Mean Similarity**: Average relevance
- **Min Similarity**: Lowest retrieved score
- **Standard Deviation**: Score consistency

## Educational Goals

- Understanding RAG architecture
- Implementing semantic search
- Using pre-trained embeddings
- Evaluating retrieval quality
- Cosine similarity applications

## Extensions

Possible enhancements:
- Add more documents
- Implement re-ranking
- Use different embedding models
- Add query expansion
- Integrate with LLM for generation

---

*Project for RAG and Information Retrieval coursework*
