import os
from collections import defaultdict

def group_by_pdf(documents):
    grouped = defaultdict(list)

    for doc in documents:
        source = doc.metadata.get("source", "")
        pdf_name = os.path.basename(source)
        grouped[pdf_name].append(doc)

    return grouped
