import gradio as gr
from dotenv import load_dotenv

from pathlib import Path

from time import perf_counter

from app.pdf_loader import load_pdf
from app.chunking import split_documents
from app.embedding import create_embedding_model
from app.vectorstore import create_vectorstore
from app.retriever import create_retriever
from app.rag import create_rag_chain
from app.llm import create_llm
from app.prompt import create_rag_prompt

from app.evaluation import run_evaluation

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

    rag_system = {
        "rag_chain":rag_chain,
        "vectorstore": vectorstore,
    }

    status = (
        f"PDF processed successfully. "
        f"Pages: {len(documents)}, Chunks: {len(chunks)}"
    )

    return rag_system, status



def ask_question(rag_system, question):
    """Ask a question using the previously created RAG chain."""
    
    if rag_system is None:
        return "Please upload a file", "","",""
    
    if not question or not question.strip():
        return "Please enter a question", "","",""

    rag_chain = rag_system["rag_chain"]
    vectorstore = rag_system["vectorstore"]

    total_start = perf_counter()

    retrieval_start = perf_counter()

    retrieved_results = (
        vectorstore.similarity_search_with_relevance_scores(
            query = question,
            k=3,
        )
    )

    retrieval_time = perf_counter() - retrieval_start

    chain_start = perf_counter()
    
    response = rag_chain.invoke({
        "input":question
    })

    chain_time = perf_counter() - chain_start
    total_time = perf_counter() - total_start
  
    answer = response["answer"]


    source_lines=[]
    seen_sources = set()
    debug_sections=[]
    scores = []


    for i, (doc, score) in enumerate(retrieved_results, start=1):
        source_path = doc.metadata.get("source", "Unknown")
        source = Path(source_path).name

        page = doc.metadata.get(
            "page_label",
            doc.metadata.get("page", "Unknown"),
        )
        scores.append(score)

        source_key = (source,page)

        if source_key not in seen_sources:
            source_lines.append(
                f"{source} - Page {page}"
            )
            seen_sources.add(source_key)

        debug_sections.append(
            f"""### Chunk {i}

**Source:** {source}

**Page:** {page}

**Relevance score:** {score:.4f}

**Content:**

{doc.page_content}
"""
        )


    sources = "\n".join(source_lines)
    retrieval_debug = "\n\n---\n\n".join(debug_sections)

    if scores:
        best_score = max(scores)
        worst_score = min(scores)
        average_score = sum(scores) / len(scores)
    else:
        best_score = 0.0
        worst_score = 0.0
        average_score = 0.0

    statistics = f"""### Retrieval Statistics

**Top-k:** 3

**Retrieved chunks:** {len(retrieved_results)}

**Best score:** {best_score:.4f}

**Average score:** {average_score:.4f}

**Worst score:** {worst_score:.4f}

### Timing

**Scored retrieval:** {retrieval_time:.4f} seconds

**RAG chain:** {chain_time:.4f} seconds

**Total:** {total_time:.4f} seconds
"""

    return answer, sources, retrieval_debug, statistics

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

            evaluation_button = gr.Button(
                "Run Evaluation",
                )
            
            evaluation_output = gr.Textbox(
                label="Evaluation Results",
                lines=4,
                interactive=False,
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

            retrieval_debug_output = gr.Markdown(
                value="Retrieved chunks will appear here.",
                label="Retrieval Debug",
            )

            statistics_output = gr.Markdown(
                value="Retrieval statistics will appear here.",
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
            retrieval_debug_output,
            statistics_output,
        ],
    )

    evaluation_button.click(
    fn=run_evaluation,
    inputs=rag_state,
    outputs=evaluation_output,
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
            retrieval_debug_output,
            statistics_output,
        ],
    )

if __name__ == "__main__":
    demo.launch()