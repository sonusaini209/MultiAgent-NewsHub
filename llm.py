"""LLM access: one shared ChatGroq model, retries with backoff, optional structured output."""
import asyncio
import logging
import os
from typing import Any, Optional, Type

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from pydantic import BaseModel

load_dotenv()

logger = logging.getLogger("newshub.llm")

MAX_ATTEMPTS = 4
NON_RETRYABLE_STATUS = (400, 401, 403, 404, 422)


class LLMError(Exception):
    """The LLM could not give a usable answer, even after retries."""


def llm_configured() -> bool:
    return bool(os.getenv("GROQ_API_KEY"))


groq_llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY") or "missing-key",  # app.py checks the key before any call
    model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
    temperature=0.3,
    max_tokens=4096,
    timeout=60,
    max_retries=0,  # retries are handled in invoke_with_retry
)


async def invoke_with_retry(runnable: Any, messages: list) -> Any:
    last_error = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return await runnable.ainvoke(messages)
        except Exception as exc:
            last_error = exc
            if getattr(exc, "status_code", None) in NON_RETRYABLE_STATUS:
                break
            if attempt < MAX_ATTEMPTS:
                delay = min(2 ** attempt * 2, 20)
                logger.warning("LLM call failed (%s); retry %d in %ds", type(exc).__name__, attempt, delay)
                await asyncio.sleep(delay)
    raise LLMError(f"{type(last_error).__name__}: {str(last_error)[:200]}") from last_error


def build_messages(system: str, human: str, variables: dict) -> list:
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
    return prompt.format_messages(**variables)


async def call_llm(system: str, human: str, variables: dict, schema: Optional[Type[BaseModel]] = None) -> Any:
    messages = build_messages(system, human, variables)

    if schema is None:
        reply = await invoke_with_retry(groq_llm, messages)
        text = (reply.content or "").strip()
        if not text:
            raise LLMError("the model returned an empty answer")
        return text

    structured_llm = groq_llm.with_structured_output(schema, method="function_calling")
    result = await invoke_with_retry(structured_llm, messages)
    if not isinstance(result, schema):
        raise LLMError("the model returned no structured output")
    return result
