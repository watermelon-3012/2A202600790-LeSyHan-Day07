from typing import Callable

from .store import EmbeddingStore


class KnowledgeBaseAgent:
    """
    An agent that answers questions using a vector knowledge base.

    Retrieval-augmented generation (RAG) pattern:
        1. Retrieve top-k relevant chunks from the store.
        2. Build a prompt with the chunks as context.
        3. Call the LLM to generate an answer.
    """

    def __init__(self, store: EmbeddingStore, llm_fn: Callable[[str], str]) -> None:
        # TODO: store references to store and llm_fn
        self.store = store
        self.llm_fn = llm_fn

    def answer(self, question: str, top_k: int = 3) -> str:
        # TODO: retrieve chunks, build prompt, call llm_fn
        chunks = self.store.search(question, top_k)
        prompt = self._build_prompt(question, chunks)
        return self.llm_fn(prompt)

    def _build_prompt(self, question: str, chunks: list[dict]) -> str:
        """Build a prompt with the retrieved chunks as context."""
        context = "\n".join([chunk.get("content", "") for chunk in chunks])
        prompt = f"""Answer the following question using the provided context.

Context:
{context}

Question: {question}

Answer:"""
        return prompt
