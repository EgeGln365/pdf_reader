from time import perf_counter


evaluation_questions = [
    {
        "question": "How does the ACS define private health insurance?",
        "expected_page": 2,
    },
    {
        "question": "When does the ACS consider a person uninsured?",
        "expected_page": 2,
    },
]


def run_evaluation(rag_system):
    if rag_system is None:
        return "Please process a PDF first."

    total_hits = 0
    reciprocal_rank_sum = 0
    total_retrieval_time = 0

    for test in evaluation_questions:
        question = test["question"]
        expected_page = test["expected_page"]

        retrieval_start = perf_counter()

        retrieved_results = (
            rag_system["vectorstore"]
            .similarity_search_with_relevance_scores(
                query=question,
                k=3,
            )
        )

        retrieval_time = perf_counter() - retrieval_start
        total_retrieval_time += retrieval_time

        retrieved_pages = []

        for doc, score in retrieved_results:
            page = doc.metadata.get(
                "page_label",
                doc.metadata.get("page", "Unknown"),
            )
            page = int(page)


            retrieved_pages.append(page)

        page_hit = expected_page in retrieved_pages

        if page_hit:
            page_rank = retrieved_pages.index(expected_page) + 1

            total_hits += 1
            reciprocal_rank_sum += 1 / page_rank
        else:
            page_rank = None

        print("\nQuestion:", question)
        print("Expected page:", expected_page)
        print("Retrieved pages:", retrieved_pages)
        print("Page hit:", page_hit)
        print("Page rank:", page_rank)
        print(f"Retrieval time: {retrieval_time:.4f} seconds")

    number_of_questions = len(evaluation_questions)

    hit_rate = total_hits / number_of_questions
    mrr = reciprocal_rank_sum / number_of_questions

    average_retrieval_time = (
        total_retrieval_time / number_of_questions
    )

    print("\n--- Evaluation Summary ---")
    print(f"Hit Rate@3: {hit_rate:.2%}")
    print(f"MRR: {mrr:.4f}")
    print(
        f"Average retrieval latency: "
        f"{average_retrieval_time:.4f} seconds"
    )

    return (
    f"Evaluation completed!\n\n"
    f"Hit Rate@3: {hit_rate:.2%}\n"
    f"MRR: {mrr:.4f}\n"
    f"Average Retrieval Latency: "
    f"{average_retrieval_time:.4f} seconds"
)