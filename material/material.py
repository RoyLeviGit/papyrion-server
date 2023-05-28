import os
from typing import List

import pinecone
from langchain import LLMChain
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.chat_vector_db.prompts import CONDENSE_QUESTION_PROMPT
from langchain.chains.combine_documents.map_reduce import MapReduceDocumentsChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.callbacks.base import AsyncCallbackHandler

from material.material_prompt import EACH_DOC_PROMPT, COMBINE_PROMPT, DOCUMENT_PROMPT
import text_extractor


class MaterialVectorstore:
    def __init__(self, user_id):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200,
        )
        embedding = OpenAIEmbeddings()
        self.user_id = user_id

        self.index = pinecone.Index(os.environ["PINECONE_INDEX"])
        self.vectorstore = Pinecone(
            index=self.index, embedding_function=embedding.embed_query, text_key="text", namespace=self.user_id
        )

    def add_docs_from_file(self, file_path: str):
        docs = text_extractor.extract_docs(file_path)
        self.add_docs(docs)

    def add_docs(self, docs: List[Document]):
        if docs:
            sub_docs = self.text_splitter.split_documents(docs)
            self.vectorstore.add_documents(sub_docs)

    def delete_vectorstore(self):
        self.index.delete(delete_all=True, namespace=self.user_id)


class Material(MaterialVectorstore):
    def __init__(
        self, user_id, callback_handler: AsyncCallbackHandler, summarize_docs=False
    ):
        # Vectorstore
        super().__init__(user_id)

        llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0.7,
        )
        streaming_llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0.7,
            streaming=True,
            callbacks=[callback_handler],
        )

        # Chain that runs the final prompt after docs have been combined
        llm_combined_docs_chain = LLMChain(llm=streaming_llm, prompt=COMBINE_PROMPT)
        combine_results_chain = StuffDocumentsChain(
            llm_chain=llm_combined_docs_chain,
            document_prompt=DOCUMENT_PROMPT,
            document_variable_name="summaries",
        )

        if summarize_docs:
            # Chain that runs on each prompt to summarise it for the context
            llm_each_doc_chain = LLMChain(llm=llm, prompt=EACH_DOC_PROMPT)
            combine_document_chain = MapReduceDocumentsChain(
                llm_chain=llm_each_doc_chain,
                combine_document_chain=combine_results_chain,
                document_variable_name="context",
            )
        else:
            combine_document_chain = combine_results_chain

        # Enveloping QA chain that runs on the previous chains
        question_generator = LLMChain(llm=llm, prompt=CONDENSE_QUESTION_PROMPT)
        self.qa_chain = ConversationalRetrievalChain(
            retriever=self.vectorstore.as_retriever(),
            question_generator=question_generator,
            combine_docs_chain=combine_document_chain,
            return_source_documents=True,
        )

    async def ask_docs(self, question: Document, chat_history=None):
        try:
            result = await self.qa_chain.acall(
                {"question": question.page_content, "chat_history": chat_history}
            )
        except Exception as e:
            print(e)
            raise

        source_documents = result["source_documents"]
        result["source_documents"] = [
            doc.metadata["source"]
            for doc in source_documents
            if "source" in doc.metadata
        ]
        return (
            question,
            result["answer"],
            result["source_documents"],
        )
