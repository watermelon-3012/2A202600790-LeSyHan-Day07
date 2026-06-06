from __future__ import annotations

from typing import Any, Callable

from .chunking import _dot, compute_similarity
from .embeddings import _mock_embed
from .models import Document


class EmbeddingStore:
    """
    A vector store for text chunks.

    Tries to use ChromaDB if available; falls back to an in-memory store.
    The embedding_fn parameter allows injection of mock embeddings for tests.
    """

    def __init__(
        self,
        collection_name: str = "documents",
        embedding_fn: Callable[[str], list[float]] | None = None,
    ) -> None:
        self._embedding_fn = embedding_fn or _mock_embed
        self._collection_name = collection_name
        self._use_chroma = False
        self._store: list[dict[str, Any]] = []
        self._collection = None
        self._next_index = 0

        try:
            import chromadb  # noqa: F401

            # TODO: initialize chromadb client + collection
            client = chromadb.Client()
            self._collection = client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            self._use_chroma = True
        except Exception:
            self._use_chroma = False
            self._collection = None

    def _make_record(self, doc: Document) -> dict[str, Any]:
        # TODO: build a normalized stored record for one document
        embedding = self._embedding_fn(doc.content)
        return {
            "id": str(self._next_index),
            "doc_id": doc.id,
            "content": doc.content,
            "embedding": embedding,
            "metadata": {"doc_id": doc.id, **doc.metadata}
        }

    def _search_records(self, query: str, records: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        # TODO: run in-memory similarity search over provided records
        query_embedding = self._embedding_fn(query)
        
        # Compute similarities for all records
        scored_records = []
        for record in records:
            similarity = compute_similarity(query_embedding, record["embedding"])
            scored_records.append({**record, "score": similarity})
        
        # Sort by score descending and return top_k
        scored_records.sort(key=lambda x: x["score"], reverse=True)
        return scored_records[:top_k]

    def add_documents(self, docs: list[Document]) -> None:
        """
        Embed each document's content and store it.

        For ChromaDB: use collection.add(ids=[...], documents=[...], embeddings=[...])
        For in-memory: append dicts to self._store
        """
        # TODO: embed each doc and add to store
        for doc in docs:
            if self._use_chroma:
                embedding = self._embedding_fn(doc.content)
                self._collection.add(
                    ids=[doc.id],
                    documents=[doc.content],
                    embeddings=[embedding],
                    metadatas=[{"doc_id": doc.id, **doc.metadata}]
                )
            else:
                record = self._make_record(doc)
                self._store.append(record)
                self._next_index += 1

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """
        Find the top_k most similar documents to query.

        For in-memory: compute dot product of query embedding vs all stored embeddings.
        """
        # TODO: embed query, compute similarities, return top_k
        if self._use_chroma:
            query_embedding = self._embedding_fn(query)
            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k
            )
            # Convert chroma results to our format
            output = []
            if results["ids"] and results["ids"][0]:
                for i, doc_id in enumerate(results["ids"][0]):
                    output.append({
                        "id": doc_id,
                        "content": results["documents"][0][i],
                        "score": results["distances"][0][i],  # chroma returns distances, not similarities
                        "metadata": results["metadatas"][0][i]
                    })
            return output
        else:
            return self._search_records(query, self._store, top_k)

    def get_collection_size(self) -> int:
        """Return the total number of stored chunks."""
        # TODO
        if self._use_chroma:
            return self._collection.count()
        else:
            return len(self._store)

    def search_with_filter(self, query: str, top_k: int = 3, metadata_filter: dict = None) -> list[dict]:
        """
        Search with optional metadata pre-filtering.

        First filter stored chunks by metadata_filter, then run similarity search.
        """
        # TODO: filter by metadata, then search among filtered chunks
        if metadata_filter is None:
            return self.search(query, top_k)
        
        if self._use_chroma:
            # For chroma, use the where clause for filtering
            query_embedding = self._embedding_fn(query)
            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=metadata_filter
            )
            output = []
            if results["ids"] and results["ids"][0]:
                for i, doc_id in enumerate(results["ids"][0]):
                    output.append({
                        "id": doc_id,
                        "content": results["documents"][0][i],
                        "score": results["distances"][0][i],
                        "metadata": results["metadatas"][0][i]
                    })
            return output
        else:
            # For in-memory, filter records first
            filtered_records = []
            for record in self._store:
                match = True
                for key, value in metadata_filter.items():
                    if record.get("metadata", {}).get(key) != value:
                        match = False
                        break
                if match:
                    filtered_records.append(record)
            
            return self._search_records(query, filtered_records, top_k)

    def delete_document(self, doc_id: str) -> bool:
        """
        Remove all chunks belonging to a document.

        Returns True if any chunks were removed, False otherwise.
        """
        # TODO: remove all stored chunks where metadata['doc_id'] == doc_id
        if self._use_chroma:
            try:
                self._collection.delete(where={"doc_id": doc_id})
                return True
            except Exception:
                return False
        else:
            initial_size = len(self._store)
            self._store = [r for r in self._store if r.get("metadata", {}).get("doc_id") != doc_id]
            return len(self._store) < initial_size
