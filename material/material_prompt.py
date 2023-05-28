from langchain import PromptTemplate

each_doc_prompt_template = """Read the document to reply the text relevant to the prompt. 
Return the relevant text verbatim.
DOCUMENT: {context}
PROMPT: {question}
RELEVANT:"""
EACH_DOC_PROMPT = PromptTemplate(
    template=each_doc_prompt_template, input_variables=["context", "question"]
)

combine_prompt_template = """Given the following context and a prompt, create a final reply with references ("SOURCES"). 
Answer in the best way you can all the question in the PROMPT. You can ask for clarifications if needed.

=========
CONTEXT:
{summaries}
=========
PROMPT:
{question}
=========
The CONTEXT was hidden from the PROMPT giver, you can use the it only if it helps.
You won't always need the CONTEXT, you can answer on your own as well. 
Do NOT state your strategy, or comment about the prompt or context, ONLY GIVE THE REPLY!
Now, complete all the tasks and answer all the questions in the PROMPT in the REPLY the step by step with explanation.
=========
REPLY:"""
COMBINE_PROMPT = PromptTemplate(
    template=combine_prompt_template, input_variables=["summaries", "question"]
)

DOCUMENT_PROMPT = PromptTemplate(
    template="CONTENT: {page_content}\nSOURCE: {source}",
    input_variables=["page_content", "source"],
)
