import gradio as gr
from dotenv import load_dotenv

from app.pdf_loader import load_pdf
from app.chunking import split_documents
from app.embedding import create_embedding_model
from app.vectorstore import create_vectorstore
from app.retriever import create_retriever
from app.rag import create_rag_chain
from app.llm import create_llm
from app.prompt import create_rag_prompt

load_dotenv(override=True)

def process_pdf(pdf_file):
    if pdf_file is None:
        return None, "Please upload a pdf"
    
    documents = load_pdf(pdf_file)
    chunks = split_documents(documents)

    embeddings = create_embedding_model()
    vectorstore = create_vectorstore(chunks, embeddings)
    retriever = create_retriever(vectorstore)

    llm = create_llm()
    prompt = create_rag_prompt()
    rag_chain = create_rag_chain(
        retriever=retriever,
        llm=llm,
        prompt=prompt,
    )

    status = (
        f"PDF processed successfully. "
        f"Pages: {len(documents)}, Chunks: {len(chunks)}"
    )

    return rag_chain, status



def ask_question(rag_chain, question):
    """Ask a question using the previously created RAG chain."""
    
    if rag_chain is None:
        return "Please upload a file", ""
    
    if not question or not question.strip():
        return "Please enter a question", ""
    
    response = rag_chain.invoke({
        "input":question
    })

    answer = response["answer"]

    source_lines=[]

    for i, doc in enumerate(response["context"],start=1):
        source= doc.metadata.get("source","Unknown")

        page = doc.metadata.get(
            "page_label",
            doc.metadata.get("page","Unknown")
        )

        source_lines.append(
            f"{i}. {source} - Page{page}"
        )

    sources = "\n".join(source_lines)

    return answer, sources
    

with gr.Blocks(title="PDF RAG CHATBOT") as demo:

    gr.Markdown(
        """
# PDF RAG Chatbot

Upload a PDF, process it, and ask questions based only on its content.
"""
    )
    rag_state = gr.State(value=None)

    with gr.Row():

        with gr.Column(scale=1):

            pdf_input = gr.File(
                label="Upload PDF",
                file_types=[".pdf"],
                type="filepath",
            )

            process_button = gr.Button(
                "Process PDF",
                variant="primary",
            )

            status_output = gr.Textbox(
                label="Status",
                interactive=False,
            )

        with gr.Column(scale=2):

            question_input = gr.Textbox(
                label="Question",
                placeholder="Ask a question about the uploaded PDF...",
                lines=3,
            )

            ask_button = gr.Button(
                "Ask Question",
                variant="primary",
            )

            answer_output = gr.Textbox(
                label="Answer",
                lines=8,
                interactive=False,
            )

            sources_output = gr.Textbox(
                label="Sources",
                lines=5,
                interactive=False,
            )

    process_button.click(
        fn=process_pdf,
        inputs=pdf_input,
        outputs=[
            rag_state,
            status_output,
        ],
    )

    ask_button.click(
        fn=ask_question,
        inputs=[
            rag_state,
            question_input,
        ],
        outputs=[
            answer_output,
            sources_output,
        ],
    )

    question_input.submit(
        fn=ask_question,
        inputs=[
            rag_state,
            question_input,
        ],
        outputs=[
            answer_output,
            sources_output,
        ],
    )

if __name__ == "__main__":
    demo.launch()