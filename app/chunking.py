from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_documents(
        documents,
        chunk_size: int=700,
        chunk_overlap: int=50
):
    """Split page-level documents into smaller chunks"""

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = text_splitter.split_documents(documents)

    return chunks
