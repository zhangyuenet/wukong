# Copyright 2026 Yue Zhang
# SPDX-License-Identifier: Apache-2.0
"""
Shared Claude API plumbing: credentials, the client and one structured call.

The API key comes from the environment (ANTHROPIC_API_KEY) or from the
repository's .env file, which is never committed.
API 密钥来自环境变量 ANTHROPIC_API_KEY，或仓库根目录的 .env（不提交）。
"""

import os
from typing import Any, List, Optional, Type

DEFAULT_MODEL = 'claude-opus-5-5'
# Server-side fallback: if a safety classifier declines, the API retries the
# request on Anthropic's recommended fallback model instead of returning a refusal.
# 服务端回退：安全分类器拒绝时，API 自动换推荐的备用模型重试。
FALLBACK_BETA = 'server-side-fallback-2026-07-01'


class LlmError(RuntimeError):
    """The model could not produce a usable answer (network, refusal, bad output)."""


def load_dotenv(path: Optional[str] = None) -> None:
    """Load KEY=VALUE lines from .env into os.environ, without overriding."""
    if path is None:
        root = os.environ.get('WUKONG_ROOT')
        if not root:
            return
        path = os.path.join(root, '.env')
    if not os.path.isfile(path):
        return
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            key, value = line.split('=', 1)
            key, value = key.strip(), value.strip().strip('"').strip("'")
            if value and key not in os.environ:
                os.environ[key] = value


def make_client(timeout: float = 60.0, max_retries: int = 2):
    """Return an Anthropic client; raise LlmError with a hint if there are no credentials."""
    load_dotenv()
    import anthropic   # imported here so offline runs do not need the package
    if not (os.environ.get('ANTHROPIC_API_KEY') or os.environ.get('ANTHROPIC_AUTH_TOKEN')):
        raise LlmError('no Claude API key: set ANTHROPIC_API_KEY in the environment or in '
                       f'{os.environ.get("WUKONG_ROOT", "<repo>")}/.env, or use --offline')
    return anthropic.Anthropic(timeout=timeout, max_retries=max_retries)


def parse(client, model: str, system: str, content: List[Any], schema: Type,
          effort: str = 'medium', max_tokens: int = 16000):
    """
    One request with a structured answer; returns the parsed pydantic object.

    Raises LlmError for refusals, truncation and API errors, so that callers
    have a single failure path (plan: do not take off; judge: no verdict).
    所有失败都转换成 LlmError，调用方只需处理一种失败。
    """
    import anthropic
    try:
        response = client.beta.messages.parse(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[{'role': 'user', 'content': content}],
            output_format=schema,
            output_config={'effort': effort},
            betas=[FALLBACK_BETA],
            fallbacks='default',
        )
    except anthropic.AuthenticationError as e:
        raise LlmError(f'Claude API rejected the key: {e.message}') from e
    except anthropic.RateLimitError as e:
        raise LlmError(f'Claude API rate limit: {e.message}') from e
    except anthropic.APIStatusError as e:
        raise LlmError(f'Claude API error {e.status_code}: {e.message}') from e
    except anthropic.APIConnectionError as e:
        raise LlmError(f'cannot reach the Claude API: {e}') from e

    if response.stop_reason == 'refusal':
        details = getattr(response, 'stop_details', None)
        raise LlmError(f'model declined the request ({getattr(details, "category", None)})')
    if response.stop_reason == 'max_tokens':
        raise LlmError('model answer was cut off (max_tokens)')
    parsed = getattr(response, 'parsed_output', None)
    if parsed is None:
        raise LlmError(f'model gave no structured answer (stop_reason {response.stop_reason})')
    return parsed
