def create_retriever(
    vectorstore,
    k: int = 3,
    search_type: str = "similarity",
):
    """Create and return a retriever from the vectorstore."""

    retriever = vectorstore.as_retriever(
        search_type=search_type,
        search_kwargs={"k": k},
    )

    return retriever