"""LLM-backed decision intelligence overlay."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any, Dict, Optional

from openai import OpenAI

logger = logging.getLogger("crisis_env.agent_assist")

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or os.getenv("GROQ_API_KEY")

if not API_KEY:
    logger.warning(
        "Agent assist missing API token. Set HF_TOKEN, API_KEY, or GROQ_API_KEY."
    )
    API_KEY = "MISSING_TOKEN"

_client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

_SYSTEM_PROMPT = (
    "You are a crisis response analyst. Evaluate the action for the current "
    "state. Respond ONLY as JSON with keys: verdict, rationale, better_action, risks. "
    "Keep rationale under 4 sentences."
)


def _build_user_prompt(payload: Dict[str, Any]) -> str:
    state = payload.get("state")
    action = payload.get("action")
    reward = payload.get("reward")
    info = payload.get("info")
    return (
        "State JSON:\n"
        f"{json.dumps(state, ensure_ascii=True)}\n\n"
        "Action JSON:\n"
        f"{json.dumps(action, ensure_ascii=True)}\n\n"
        f"Reward: {reward}\n"
        f"Info: {json.dumps(info, ensure_ascii=True)}\n"
    )


def _sync_explain(payload: Dict[str, Any]) -> Dict[str, Any]:
    if API_KEY == "MISSING_TOKEN":
        return {
            "error": "Missing API token. Set HF_TOKEN, API_KEY, or GROQ_API_KEY.",
        }

    prompt = _build_user_prompt(payload)
    response = _client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=220,
    )
    content = response.choices[0].message.content or "{}"
    try:
        return json.loads(content)
    except Exception:
        return {
            "verdict": "unknown",
            "rationale": content.strip()[:600],
            "better_action": None,
            "risks": None,
        }


async def explain(payload: Dict[str, Any]) -> Dict[str, Any]:
    return await asyncio.to_thread(_sync_explain, payload)
