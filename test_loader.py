from app.pdf_loader import load_pdf
from app.chunking import split_documents
from app.embedding import create_embedding_model
from app.vectorstore import create_vectorstore
from app.retriever import create_retriever

documents = load_pdf("us_census/acsbr-015.pdf")
chunks = split_documents(documents)

embeddings = create_embedding_model()
vectorstore = create_vectorstore(chunks, embeddings)
retriever = create_retriever(vectorstore, k=3)

query = "What is health insurance coverage?"
results = retriever.invoke(query)

print(f"Pages loaded: {len(documents)}")
print(f"Chunks created: {len(chunks)}")
print(f"Retrieved chunks: {len(results)}")

for i, doc in enumerate(results, start=1):
    print(f"\nResult {i}")
    print(f"Page: {doc.metadata.get('page_label', doc.metadata.get('page'))}")
    print(doc.page_content[:400])