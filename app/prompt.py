from langchain_core.prompts import ChatPromptTemplate


def create_rag_prompt():
    """Create and return the RAG prompt."""

    prompt = ChatPromptTemplate.from_template(
        """
Use the following context to answer the user's question.

Rules:
- Answer only using information found in the context.
- If the answer is not available in the context, say that you do not know.
- Keep the answer clear and concise.

Context:
{context}

Question:
{input}
"""
    )

    return prompt