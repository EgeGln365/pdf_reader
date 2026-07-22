from langchain_community.vectorstores import FAISS

def create_vectorstore(chunks,embeddings):
    """Create FAISS vector store from document chunks"""

    vectorstore = FAISS.from_documents(
        chunks,
        embeddings
    )

    return vectorstore