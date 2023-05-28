# Papyrion Server

Papyrion Server is the papyrion application server developed using FastAPI for serving various endpoints related to file handling, user authentication, and material handling.

See: `https://github.com/RoyLeviGit/papyrion`

## Prerequisites

Before you begin, ensure you have met the following requirements:

- You have installed Python 3.8 or higher
- You have installed FastAPI and the necessary libraries (listed in `requirements.txt`)
- You have Pinecone, OpenAI, and Translator Text API keys

## Environment Variables

Create a `.env` file in the root of the application using the `.env.template` as a guide.

```
JWT_ACCESS_SECRET=<your JWT access secret>
JWT_REFRESH_SECRET=<your JWT refresh secret>

OPENAI_API_KEY=<your OpenAI API key>

PINECONE_API_KEY=<your Pinecone API key>
PINECONE_ENV=<your Pinecone environment>
PINECONE_INDEX=<your Pinecone index>

TRANSLATOR_TEXT_SUBSCRIPTION_KEY=<your Translator Text subscription key>
TRANSLATOR_TEXT_REGION=<your Translator Text region>
TRANSLATOR_TEXT_ENDPOINT=<your Translator Text endpoint>
```

Replace `<your ...>` with your actual data.

## Running Papyrion Server

To start the Papyrion Server, navigate to the root directory of the project and run the following command:

```
python app.py
```

The server will start and listen on `http://localhost:8000`.

## API Endpoints

Here are some of the core endpoints provided by Papyrion Server:

- `/auth`: Generates JWT access and refresh tokens for a user.
- `/refresh`: Refreshes the JWT access and refresh tokens.
- `/upload`: Uploads a file and returns a document ID.
- `/delete-files`: Deletes all files for a user.
- `/list-files`: Lists all files for a user.
- `/question_doc`: Answers questions about a document.
- `/completion`: Provides chat completion.
