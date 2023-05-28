from langchain import PromptTemplate

QUESTION_WRAPPER = "###QQQ###"
CONTEXT_WRAPPER = "###CCC###"
NONE_WRAPPER = "###NO_LIST###"
each_doc_prompt_template = f"""\
You are a world-class document analysis tool designed to identify questions and tasks in a provided text. 

You should always analyze the smallest question or task possible, refrain from creating large questions or to-do tasks.
The question may contain small hints.

Use {QUESTION_WRAPPER} to enclose each individual question or task.

For example:
\"\"\"
TEXT:
Some text extracted from a file

AI:
{QUESTION_WRAPPER} <Action (prove, answer, do, etc.)>: <VERBATIM question or task text> {QUESTION_WRAPPER}
{QUESTION_WRAPPER} <Action (prove, answer, do, etc.)>: <VERBATIM question or task text> {QUESTION_WRAPPER}
{QUESTION_WRAPPER} <Action (prove, answer, do, etc.)>: <VERBATIM question or task text> {QUESTION_WRAPPER}
...
\"\"\"

There won't always be questions or tasks. If there are none, respond with:
\"\"\"
{NONE_WRAPPER}
\"\"\"

REMEMBER(!), repeat the text VERBATIM.
TEXT:
{{text}}

AI:
"""
QUESTION_PROMPT = PromptTemplate(
    template=each_doc_prompt_template, input_variables=["text"]
)
