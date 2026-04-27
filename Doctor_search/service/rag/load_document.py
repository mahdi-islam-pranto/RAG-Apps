import json
from typing import List, Dict

# load the doctors document from the local system
JSON_PATH = "documents/doctors.json"

def load_doctor_documents(json_path: str) -> List[Dict]:
    """
    Load doctors from JSON file and convert them into searchable documents.
    
    Returns:
        List of dicts with:
        - doc_id
        - content (for embedding / RAG)
        - metadata
    """

    # load the doctors document from the local system
    with open(json_path, "r", encoding="utf-8") as f:
        doctors = json.load(f)

    # create a list to store the documents
    documents = []

    for doctor in doctors:
        # Basic validation
        if not all(k in doctor for k in ("doctor_id", "name", "department")):
            continue

        # create the content for the document
        content = (
            f"{doctor['name']} is a doctor specializing in "
            f"{doctor['department']}."
        )

        doc = {
            "doc_id": doctor["doctor_id"],
            "content": content,
            "metadata": {
                "name": doctor["name"],
                "department": doctor["department"]
            }
        }

        documents.append(doc)

    return documents

# test the function
# if __name__ == "__main__":
#     documents = load_doctor_documents(JSON_PATH)
#     print(documents)
