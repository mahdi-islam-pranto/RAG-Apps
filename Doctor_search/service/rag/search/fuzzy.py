from rapidfuzz import process, fuzz


def fuzzy_retriever(query, lc_documents, k=5, threshold=65):
    choices = [
        f"{doc.metadata['name']} {doc.metadata['department']}"
        for doc in lc_documents
    ]

    matches = process.extract(
        query,
        choices,
        scorer=fuzz.WRatio,
        limit=k
    )

    results = []
    for _, score, idx in matches:
        if score >= threshold:
            results.append(lc_documents[idx])

    return results
