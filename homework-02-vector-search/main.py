# LLM Zoomcamp HW2 - Vector Search

import sys
import numpy as np

sys.path.insert(0, "/workspaces/llm-zoomcamp-2026-code/homework-02-vector-search")

from embedder import Embedder
from gitsource import GithubRepositoryDataReader, chunk_documents
from minsearch import Index, VectorSearch

embedder = Embedder(
    path="/workspaces/llm-zoomcamp-2026-code/homework-02-vector-search/models/Xenova/all-MiniLM-L6-v2"
)

# Q1
v = embedder.encode("How does approximate nearest neighbor search work?")
print("Q1 (v[0]):", round(float(v[0]), 2))

# pull the lessons at the pinned commit so everyone has the same data
reader = GithubRepositoryDataReader(
    repo_owner="DataTalksClub",
    repo_name="llm-zoomcamp",
    commit_id="8c1834d",
    allowed_extensions={"md"},
    filename_filter=lambda path: "/lessons/" in path,
)
documents = [file.parse() for file in reader.read()]

# Q2 - vectors are normalized so dot product == cosine similarity
target = next(d for d in documents if d["filename"] == "02-vector-search/lessons/07-sqlitesearch-vector.md")
target_vec = embedder.encode(target["content"])
print("Q2 (cosine similarity):", round(float(np.dot(v, target_vec)), 2))

# Q3
chunks = chunk_documents(documents, size=2000, step=1000)
X = embedder.encode_batch([c["content"] for c in chunks])
scores = X.dot(v)
best_idx = int(np.argmax(scores))
print("Q3 (highest-scoring chunk):", chunks[best_idx]["filename"])

# Q4
vector_index = VectorSearch(keyword_fields=["filename"])
vector_index.fit(X, chunks)

q4_vec = embedder.encode("What metric do we use to evaluate a search engine?")
print("Q4 (first result):", vector_index.search(q4_vec, num_results=5)[0]["filename"])

# Q5 - compare vector vs text search
text_index = Index(text_fields=["content"], keyword_fields=["filename"])
text_index.fit(chunks)

q5_vec = embedder.encode("How do I store vectors in PostgreSQL?")
q5_vector_results = vector_index.search(q5_vec, num_results=5)
q5_text_results = text_index.search("How do I store vectors in PostgreSQL?", num_results=5)

only_in_vector = {r["filename"] for r in q5_vector_results} - {r["filename"] for r in q5_text_results}
print("Q5 (in vector but not text):", only_in_vector)

# Q6 - hybrid search with RRF
def rrf(result_lists, k=60, num_results=5):
    scores = {}
    docs = {}
    for results in result_lists:
        for rank, doc in enumerate(results):
            key = (doc["filename"], doc["start"])
            scores[key] = scores.get(key, 0) + 1 / (k + rank)
            docs[key] = doc
    ranked = sorted(scores, key=scores.get, reverse=True)
    return [docs[key] for key in ranked[:num_results]]

q6_vec = embedder.encode("How do I give the model access to tools?")
q6_results = rrf([
    vector_index.search(q6_vec, num_results=5),
    text_index.search("How do I give the model access to tools?", num_results=5),
])
print("Q6 (first after RRF):", q6_results[0]["filename"])
