import chromadb
from chromadb.config import Settings
from openai import OpenAI


class FinancialSituationMemory:
    def __init__(self, name, config):
        self.config = config
        self.llm_provider = (config.get("llm_provider") or "").lower()
        self.is_nvidia_backend = "nvidia.com" in (config.get("backend_url") or "")

        # Resolve embeddings base URL
        explicit_embed_url = config.get("embeddings_backend_url")
        if explicit_embed_url:
            self.embeddings_base_url = explicit_embed_url
        elif config["backend_url"] == "http://localhost:11434/v1":
            self.embeddings_base_url = config["backend_url"]
        elif self.llm_provider == "nvidia" or self.is_nvidia_backend:
            # Use local Ollama for embeddings when NVIDIA is used for chat
            self.embeddings_base_url = "http://localhost:11434/v1"
        else:
            self.embeddings_base_url = config["backend_url"]

        # Resolve embeddings model
        explicit_embed_model = config.get("embeddings_model")
        if explicit_embed_model:
            self.embedding_model = explicit_embed_model
        else:
            if self.embeddings_base_url == "http://localhost:11434/v1":
                self.embedding_model = "nomic-embed-text"
            elif (self.llm_provider == "nvidia" or self.is_nvidia_backend) and "nvidia.com" in self.embeddings_base_url:
                self.embedding_model = "snowflake/arctic-embed-l"
            else:
                self.embedding_model = "text-embedding-3-small"

        # Create separate clients for chat and embeddings
        self.client = OpenAI(base_url=config["backend_url"])
        self.embeddings_client = OpenAI(base_url=self.embeddings_base_url)

        # Vector DB for memory
        self.chroma_client = chromadb.Client(Settings(allow_reset=True))
        self.situation_collection = self.chroma_client.create_collection(name=name)

    def get_embedding(self, text: str, input_type: str = "passage"):
        """Get an embedding for text.

        input_type: "query" for search queries, "passage" for stored texts.
        """
        if "nvidia.com" in self.embeddings_base_url:
            # NVIDIA Retrieval API (only used when explicitly configured)
            response = self.embeddings_client.embeddings.create(
                input=[text],
                model=self.embedding_model,
                encoding_format="float",
                extra_body={"input_type": input_type, "truncate": "NONE"},
            )
        else:
            # OpenAI-compatible embeddings (OpenAI or Ollama)
            response = self.embeddings_client.embeddings.create(
                model=self.embedding_model,
                input=text,
            )
        return response.data[0].embedding

    def add_situations(self, situations_and_advice):
        """Add financial situations and their corresponding advice. Parameter is a list of tuples (situation, rec)."""

        situations = []
        advice = []
        ids = []
        embeddings = []

        offset = self.situation_collection.count()

        for i, (situation, recommendation) in enumerate(situations_and_advice):
            situations.append(situation)
            advice.append(recommendation)
            ids.append(str(offset + i))
            embeddings.append(self.get_embedding(situation, input_type="passage"))

        self.situation_collection.add(
            documents=situations,
            metadatas=[{"recommendation": rec} for rec in advice],
            embeddings=embeddings,
            ids=ids,
        )

    def get_memories(self, current_situation, n_matches=1):
        """Find matching recommendations using vector similarity search."""
        query_embedding = self.get_embedding(current_situation, input_type="query")

        results = self.situation_collection.query(
            query_embeddings=[query_embedding],
            n_results=n_matches,
            include=["metadatas", "documents", "distances"],
        )

        matched_results = []
        for i in range(len(results["documents"][0])):
            matched_results.append(
                {
                    "matched_situation": results["documents"][0][i],
                    "recommendation": results["metadatas"][0][i]["recommendation"],
                    "similarity_score": 1 - results["distances"][0][i],
                }
            )

        return matched_results


if __name__ == "__main__":
    # Example usage skipped in production
    pass
