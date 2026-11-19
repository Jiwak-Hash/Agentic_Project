"""One document per short policy; normalized embeddings give cosine similarity."""

import hashlib
import json
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer

PROJECT_DIR = Path(__file__).resolve().parents[1]
POLICY_DIR = PROJECT_DIR / "data" / "policies"
INDEX_DIR = PROJECT_DIR / "data" / "faiss_index"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class PolicyRetriever:
    def __init__(self, policy_dir=POLICY_DIR, index_dir=INDEX_DIR):
        self.policy_dir = Path(policy_dir)
        self.index_dir = Path(index_dir)
        paths = sorted(self.policy_dir.glob("*.md"))
        if not paths:
            raise ValueError(f"No Markdown policy files found in {self.policy_dir}")
        self.documents = [
            {"source": path.name, "text": path.read_text(encoding="utf-8").strip()}
            for path in paths
        ]
        if any(not document["text"] for document in self.documents):
            raise ValueError("Policy files must not be empty.")
        fingerprint = hashlib.sha256(
            json.dumps([MODEL_NAME, self.documents], sort_keys=True).encode()
        ).hexdigest()
        # Download once if needed; inference always runs locally on CPU.
        self.model = SentenceTransformer(
            MODEL_NAME, device="cpu", cache_folder=str(PROJECT_DIR / "data" / "model_cache")
        )
        # Keep each policy short enough to embed without silently losing text.
        for document in self.documents:
            if len(self.model.tokenizer.encode(document["text"])) > self.model.max_seq_length:
                raise ValueError(f"Split the long policy into shorter files: {document['source']}")
        index_path = self.index_dir / "policies.faiss"
        metadata_path = self.index_dir / "metadata.json"
        self.index = None
        if index_path.exists() and metadata_path.exists():
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                if metadata.get("fingerprint") == fingerprint:
                    candidate = faiss.read_index(str(index_path))
                    if (candidate.ntotal == len(self.documents)
                            and candidate.d == self.model.get_embedding_dimension()):
                        self.index = candidate
            except (OSError, ValueError, RuntimeError):
                pass  # Rebuild a missing, outdated, or damaged cache.
        self.rebuilt = self.index is None
        if self.rebuilt:
            vectors = self.model.encode(
                [document["text"] for document in self.documents],
                normalize_embeddings=True, convert_to_numpy=True,
            ).astype("float32")
            self.index = faiss.IndexFlatIP(vectors.shape[1])
            self.index.add(vectors)
            self.index_dir.mkdir(parents=True, exist_ok=True)
            faiss.write_index(self.index, str(index_path))
            metadata_path.write_text(json.dumps({
                "fingerprint": fingerprint, "model": MODEL_NAME,
                "documents": self.documents,
            }, indent=2), encoding="utf-8")

    def retrieve(self, question, top_k=3):
        if not isinstance(question, str) or not question.strip():
            raise ValueError("Please enter a non-empty customer question.")
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")
        vector = self.model.encode(
            [question], normalize_embeddings=True, convert_to_numpy=True
        ).astype("float32")
        scores, indices = self.index.search(vector, min(top_k, len(self.documents)))
        return [dict(self.documents[int(index)], score=float(score))
                for score, index in zip(scores[0], indices[0]) if index >= 0]


if __name__ == "__main__":
    try:
        retriever = PolicyRetriever()
        print(f"{'Built' if retriever.rebuilt else 'Loaded'} index: {INDEX_DIR}")
        print(f"Policies: {len(retriever.documents)}")
    except Exception as error:
        raise SystemExit(f"Could not prepare policy retrieval: {error}") from error
