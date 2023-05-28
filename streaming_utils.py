import json

import asyncio
from typing import Any, Dict, List, Optional
from uuid import UUID

from langchain.callbacks.base import AsyncCallbackHandler
from langchain.schema import LLMResult


class Stream:
    def __init__(self) -> None:
        self._queue = asyncio.Queue[dict]()

    def __aiter__(self) -> "Stream":
        return self

    async def __anext__(self) -> dict:
        return await self._queue.get()

    async def asend(self, value: dict) -> None:
        await self._queue.put(value)


class ExplicitAsyncCallbackHandler(AsyncCallbackHandler):
    async def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        **kwargs: Any,
    ) -> None:
        print({"serialized": serialized, "prompts": prompts, **kwargs})

    async def on_llm_end(
            self,
            response: LLMResult,
            **kwargs: Any
    ) -> None:
        print({"response": response, **kwargs})

    async def on_end(self) -> None:
        pass

class NonFilteredAsyncCallbackHandler(ExplicitAsyncCallbackHandler):
    def __init__(self, stream: Stream):
        self.stream = stream

    async def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        await self.stream.asend({"event": f"new_token", "data": {"token": token}})

    async def on_end(self) -> None:
        """Run when the LLM finishes generating."""
        await self.stream.asend({"event": "end_stream"})


class QuestionFilteredAsyncCallbackHandler(ExplicitAsyncCallbackHandler):
    def __init__(self, stream: Stream, delimiter: str = "###QQQ###"):
        self.stream = stream
        self.delimiter = delimiter
        self.delimiter_index = 0
        self.send_tokens = False
        self.token_buffer = ""

    async def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        for char in token:
            # Check if the current character matches the delimiter at the current index
            if char == self.delimiter[self.delimiter_index]:
                self.delimiter_index += 1

                # If the entire delimiter is matched
                if self.delimiter_index == len(self.delimiter):
                    self.delimiter_index = 0
                    self.send_tokens = not self.send_tokens

                    # If send_tokens is False after toggling and token_buffer has content, signal the end of a block.
                    # Send the content with the delimiter flag set to True
                    if not self.send_tokens:
                        await self.stream.asend(
                            {
                                "event": "new_token",
                                "data": {
                                    "token": self.token_buffer,
                                    "delimiter": True,
                                },
                            }
                        )
                        self.token_buffer = ""
            else:
                # If send_tokens is True, add any partial delimiter and the current character to token_buffer
                if self.send_tokens:
                    self.token_buffer += self.delimiter[: self.delimiter_index]
                    self.token_buffer += char
                # Reset the delimiter_index since the character didn't match the delimiter
                self.delimiter_index = 0

        # If send_tokens is True and token_buffer has content, send the content
        if self.send_tokens and self.token_buffer:
            await self.stream.asend(
                {"event": "new_token", "data": {"token": self.token_buffer}}
            )
            self.token_buffer = ""

    async def on_end(self) -> None:
        """Run when the LLM finishes generating."""
        await self.stream.asend({"event": "end_stream"})
