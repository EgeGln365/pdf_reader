from langchain_openai import ChatOpenAI

def create_llm(
        model_name: str="gpt-4o-mini",
        temperature: float=0
):
    """Create and return a chat model"""

    llm = ChatOpenAI(
        model=model_name,
        temperature=temperature
    )

    return llm
