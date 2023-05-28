import re

from langchain.callbacks.base import AsyncCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain import LLMChain
from langchain.schema import Document

import text_extractor
from question.question_prompt import QUESTION_PROMPT, QUESTION_WRAPPER, CONTEXT_WRAPPER


class Question:
    def __init__(self, callback_handler: AsyncCallbackHandler):
        llm = ChatOpenAI(
            model_name="gpt-4",
            temperature=0,
            request_timeout=180,
            streaming=True,
            callbacks=[callback_handler],
        )
        self.llm_chain = LLMChain(prompt=QUESTION_PROMPT, llm=llm)

    async def get_questions_and_context(self, question_docs_path: str):
        question_docs = text_extractor.extract_docs(question_docs_path)

        for question_doc in question_docs:
            full_text = question_doc.page_content
            try:
                response = await self.llm_chain.acall({"text": full_text})
            except Exception as e:
                print(f"Got api call error: {e}")
                continue

    @staticmethod
    def _get_context_and_questions(question_doc, response):
        result = response["text"]
        # Extract context
        context_regex = rf"{CONTEXT_WRAPPER}(.*?){CONTEXT_WRAPPER}"
        context_matches = re.findall(context_regex, result, re.DOTALL)
        context = None
        if context_matches:
            context = Document(
                page_content="\n".join([c.strip() for c in context_matches]),
                metadata=question_doc.metadata,
            )

        # Extract questions
        question_regex = rf"{QUESTION_WRAPPER}(.*?){QUESTION_WRAPPER}"
        question_matches = re.findall(question_regex, result, re.DOTALL)
        questions = [
            Document(page_content=q.strip(), metadata=question_doc.metadata)
            for q in question_matches
        ]

        return  context, questions
