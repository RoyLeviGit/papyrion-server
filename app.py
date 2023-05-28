import json
from typing import List, Dict

import asyncio
import os

from fastapi import (
    FastAPI,
    Security,
    HTTPException,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
from fastapi_jwt import JwtAuthorizationCredentials, JwtAccessBearer, JwtRefreshBearer
from pydantic import BaseModel
from uuid import uuid4
import pinecone

from file_handler import FileHandler
from material import MaterialVectorstore, Material
from question import Question
from streaming_utils import (
    Stream,
    QuestionFilteredAsyncCallbackHandler,
    NonFilteredAsyncCallbackHandler,
)
from utils import (
    document_from_dict,
)

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

pinecone.init(
    api_key=os.environ["PINECONE_API_KEY"], environment=os.environ["PINECONE_ENV"]
)

file_handler = FileHandler()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Read access token from bearer header
access_security = JwtAccessBearer(
    secret_key=os.environ["JWT_ACCESS_SECRET"],
)
# Read refresh token from bearer header only
refresh_security = JwtRefreshBearer(
    secret_key=os.environ["JWT_REFRESH_SECRET"],
)


@app.post("/auth")
def auth():
    subject = {"sub": str(uuid4())}

    # Create new access/refresh tokens pair
    access_token = access_security.create_access_token(subject=subject)
    refresh_token = refresh_security.create_refresh_token(subject=subject)

    return {"access_token": access_token, "refresh_token": refresh_token}


@app.post("/refresh")
def refresh(credentials: JwtAuthorizationCredentials = Security(refresh_security)):
    # Update access/refresh tokens pair
    access_token = access_security.create_access_token(subject=credentials.subject)
    refresh_token = refresh_security.create_refresh_token(subject=credentials.subject)

    return {"access_token": access_token, "refresh_token": refresh_token}


@app.post("/upload")
async def upload_file(
    file: UploadFile,
    credentials: JwtAuthorizationCredentials = Security(access_security),
):
    user_id = credentials["sub"]

    document_id = file_handler.save_file(file, user_id)
    try:
        MaterialVectorstore(user_id).add_docs_from_file(
            file_handler.get_file(user_id, document_id)
        )
    except Exception as e:
        print(f"Exception while adding file to vectorstore: {e}")
        file_handler.delete_file(user_id, document_id)
        raise HTTPException(status_code=400, detail=f"File failed to upload: {e}")

    await file.close()
    return {"message": f"File uploaded successfully", "document_id": document_id}


@app.post("/delete-files")
def delete_files(credentials: JwtAuthorizationCredentials = Security(access_security)):
    user_id = credentials["sub"]
    file_handler.delete_user_files(user_id)
    MaterialVectorstore(user_id).delete_vectorstore()

    return {"message": f"Files deleted successfully"}


@app.get("/list-files")
def list_files(
    credentials: JwtAuthorizationCredentials = Security(access_security),
):
    user_id = credentials["sub"]
    return file_handler.list_files(user_id)


class QuestionDocBody(BaseModel):
    document_id: str


@app.post("/question_doc")
async def sse_question_doc(
    body: QuestionDocBody,
    credentials: JwtAuthorizationCredentials = Security(access_security),
) -> EventSourceResponse:
    user_id = credentials["sub"]
    stream = Stream()
    task = asyncio.ensure_future(question_doc(user_id, body, stream))

    async def event_publisher():
        try:
            async for event in stream:
                yield json.dumps(event)
        except asyncio.CancelledError as e:
            task.cancel()
            raise e

    return EventSourceResponse(event_publisher())


async def question_doc(user_id: str, body: QuestionDocBody, stream: Stream):
    document_id = body.document_id
    if not file_handler.file_exists(user_id, document_id):
        raise HTTPException(status_code=400, detail="File not found")

    question_filtered_callback = QuestionFilteredAsyncCallbackHandler(stream)
    await Question(question_filtered_callback).get_questions_and_context(
        file_handler.get_file(user_id, document_id)
    )
    await question_filtered_callback.on_end()


class CompletionBody(BaseModel):
    prompt: Dict[str, str]
    chat_history: List[str] = []


@app.post("/completion")
async def sse_completion(
    body: CompletionBody,
    credentials: JwtAuthorizationCredentials = Security(access_security),
) -> EventSourceResponse:
    user_id = credentials["sub"]
    stream = Stream()
    task = asyncio.ensure_future(completion(user_id, body, stream))

    async def event_publisher():
        try:
            async for event in stream:
                yield json.dumps(event)
        except asyncio.CancelledError as e:
            task.cancel()
            raise e

    return EventSourceResponse(event_publisher())


async def completion(user_id: str, body: CompletionBody, stream: Stream):
    prompt = body.prompt
    if not prompt:
        raise HTTPException(status_code=400, detail="Missing prompt in the request")
    prompt_doc = document_from_dict(prompt)
    chat_history = body.chat_history

    unfiltered_callback = NonFilteredAsyncCallbackHandler(stream)
    material = Material(user_id, unfiltered_callback)
    await material.ask_docs(prompt_doc, chat_history)
    await unfiltered_callback.on_end()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", reload=True)
