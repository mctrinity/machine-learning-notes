# 🚀 Guide to Embeddings & Vector Stores

## 📌 What Are Embeddings?
Embeddings are numerical vector representations of data (such as text, images, or audio) that capture semantic meaning and relationships. They allow AI models to process and compare entities efficiently.

### 🔹 Example: Word Embeddings
A word like **"king"** could be represented as:
```
[0.12, -0.45, 0.78, 0.34, -0.67, ...]  # (300-dimensional vector)
```
Words like **"queen"** and **"prince"** will have similar vectors, showing their relationship.

---

## 🏗 How to Generate Embeddings
### ✅ Using SentenceTransformers (BERT-based model)
```python
from sentence_transformers import SentenceTransformer

# Load pre-trained model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Generate embeddings for a list of texts
texts = ["Artificial intelligence is amazing!", "Machine learning is fun!", "Deep learning is powerful."]
embeddings = model.encode(texts)

print("Embedding shape:", embeddings.shape)  # (3, 384) → 3 texts, 384-dimensional vectors
```

---

## 🏛 Storing Embeddings in a Vector Database

### 🔹 **Option 1: FAISS (Local, Fast, Offline Use)**
```python
import faiss
import numpy as np

# Get embedding dimension
dim = embeddings.shape[1]  

# Create FAISS index
index = faiss.IndexFlatL2(dim)

# Convert to NumPy array and add embeddings
embeddings_np = np.array(embeddings).astype("float32")
index.add(embeddings_np)

print("Stored", index.ntotal, "embeddings in FAISS")
```
#### 🔍 Search for Similar Texts
```python
query_text = "AI is incredible!"
query_embedding = model.encode([query_text]).astype("float32")

D, I = index.search(query_embedding, k=1)
print("Most similar text:", texts[I[0][0]])
```

---

### 🔹 **Option 2: Pinecone (Cloud, Scalable, API-Based)**
```python
import pinecone

pinecone.init(api_key="YOUR_PINECONE_API_KEY", environment="us-west1-gcp")
index_name = "my-embedding-store"
if index_name not in pinecone.list_indexes():
    pinecone.create_index(index_name, dimension=384, metric="cosine")

index = pinecone.Index(index_name)
vectors = [(str(i), embeddings[i].tolist(), {}) for i in range(len(texts))]
index.upsert(vectors)

print("Stored", len(vectors), "embeddings in Pinecone")
```
#### 🔍 Search for Similar Texts
```python
query_embedding = model.encode(["AI is amazing!"]).tolist()
results = index.query(queries=query_embedding, top_k=1, include_metadata=True)
print("Most similar text:", results["matches"][0]["id"])
```

---

### 🔹 **Option 3: ChromaDB (Local & Cloud AI Apps)**
```python
import chromadb

client = chromadb.PersistentClient(path="./chroma_store")
collection = client.get_or_create_collection(name="my_collection")

for i, text in enumerate(texts):
    collection.add(ids=[str(i)], embeddings=[embeddings[i].tolist()], metadatas=[{"text": text}])

print("Stored", len(texts), "embeddings in ChromaDB")
```
#### 🔍 Search for Similar Texts
```python
query_embedding = model.encode(["AI is amazing!"]).tolist()
results = collection.query(query_embeddings=query_embedding, n_results=1)
print("Most similar text:", results["metadatas"][0]["text"])
```

---

## 🔥 Which Vector Store Should You Use?
| **Database**   | **Best for**                             | **Pros**                          | **Cons**                          |
|---------------|---------------------------------|---------------------------------|--------------------------------|
| **FAISS**     | Local, fast, small-scale       | Efficient, free, offline       | No built-in metadata storage  |
| **Pinecone**  | Large-scale, cloud-based      | Scalable, managed service      | Requires API key (paid for large data) |
| **ChromaDB**  | Local + cloud AI apps         | Simple, metadata support       | Not as optimized for billions of embeddings |

---

## 🎯 Conclusion
Embeddings are powerful representations for **search, recommendation systems, and AI-driven applications**. You can store and search them efficiently using **FAISS, Pinecone, or ChromaDB**.

🚀 Ready to build an AI-powered search system? Let’s go! 🔍
