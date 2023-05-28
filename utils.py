from langchain.schema import Document


def stream_event_from_dict(stream_event: dict) -> str:
    return f"data:{stream_event}\n\n"


def dict_from_document_list(documents: list) -> list:
    return [dict_from_document(document) for document in documents]


def dict_from_document(document: Document) -> dict:
    if document is None:
        return {}
    return {
        "page_content": document.page_content,
        "metadata": document.metadata,
    }


def document_from_dict(doc_dict: dict) -> Document:
    if doc_dict is None:
        return Document()
    if "page_content" in doc_dict and "metadata" not in doc_dict:
        return Document(page_content=doc_dict["page_content"])
    if "metadata" in doc_dict and "page_content" not in doc_dict:
        return Document(metadata=doc_dict["metadata"])
    if "metadata" in doc_dict and "page_content" in doc_dict:
        return Document(
            page_content=doc_dict["page_content"],
            metadata=doc_dict["metadata"],
        )
    return Document()
