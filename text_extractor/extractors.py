import re
import nltk
import ssl
import os

from langchain.document_loaders import (
    UnstructuredWordDocumentLoader,
    TextLoader,
    CSVLoader,
    PyPDFLoader,
    UnstructuredPowerPointLoader,
)
from langdetect import detect
from azure.ai.translation.text.models import InputTextItem
from azure.ai.translation.text import TextTranslationClient, TranslatorCredential


nltk_resources = ["punkt", "averaged_perceptron_tagger"]
for resource in nltk_resources:
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context
    nltk.download(resource)


def extract_docs(file_path):
    file_extension = file_path.rsplit(".", 1)[1].lower()
    if file_extension == "markdown":
        # use markdown loader
        pass
    elif file_extension == "html":
        # use HTML loader
        pass
    elif file_extension == "doc" or file_extension == "docx":
        # use Microsoft Word loader
        return _extract_docs(file_path, UnstructuredWordDocumentLoader)
    elif file_extension == "txt":
        # use plain text loader
        return _extract_docs(file_path, TextLoader)
    elif file_extension == "csv":
        # use CSV loader
        return _extract_docs(file_path, TextLoader)
    elif file_extension == "json":
        # use JSON loader
        pass
    elif file_extension == "xml":
        # use XML loader
        pass
    elif file_extension == "pdf":
        # use PDF loader
        return _extract_docs(file_path, PyPDFLoader)
    elif file_extension == "ppt" or file_extension == "pptx":
        # use Microsoft PowerPoint loader
        return _extract_docs(file_path, UnstructuredPowerPointLoader)

    # handle unsupported file type
    raise NotImplementedError("This file type is not yet supported")


def _extract_docs(file_path, loader, **loader_kwargs):
    file_docs = loader(file_path, **loader_kwargs).load()
    for file_doc in file_docs:
        raw_text = file_doc.page_content
        formatted_text = re.sub("\n{2,}", "\n\n", raw_text)
        if not formatted_text:
            continue
        try:
            english_text = _get_english_text(formatted_text)
        except Exception as e:
            print(f"Exception in translation: {e}")
            english_text = formatted_text

        file_doc.page_content = english_text
        file_doc.metadata["source"] = os.path.basename(file_doc.metadata["source"])
    return file_docs


def _get_english_text(text, target_language="en"):
    source_language = detect(text)
    if source_language == target_language:
        return text

    # Set up the Azure Translate client
    api_key = os.environ["TRANSLATOR_TEXT_SUBSCRIPTION_KEY"]
    api_region = os.environ["TRANSLATOR_TEXT_REGION"]
    api_endpoint = os.environ["TRANSLATOR_TEXT_ENDPOINT"]

    text_translator = TextTranslationClient(
        endpoint=api_endpoint,
        credential=TranslatorCredential(api_key, api_region)
    )

    input_text_elements = [InputTextItem(text=text)]

    # Translate the text to English
    response = text_translator.translate(
        content=input_text_elements,
        to=[target_language],
        from_parameter=source_language
    )

    return response[0].translations[0].text
