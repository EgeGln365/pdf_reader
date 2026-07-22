from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import (
    create_stuff_documents_chain,
)

def create_rag_chain(retriever, llm, prompt):
    """Create a complete RAG chain."""
    
    document_chain = create_stuff_documents_chain(
        llm,
        prompt
    )

    rag_chain = create_retrieval_chain(
        retriever,
        document_chain
    )

    return rag_chain
