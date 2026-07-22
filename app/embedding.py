from langchain_huggingface import HuggingFaceEmbeddings

def create_embedding_model():
    """Create and return the embedding model."""
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    return embeddings
    