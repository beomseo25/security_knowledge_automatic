"""LLM 호출 공통 부품 (Gemini / OpenAI / Anthropic 중 아무거나).

이 프로젝트에는 스크립트가 쓸 수 있는 LLM 키가 원래 없었다. 그래서 어느 제공자
키가 들어오든 같은 코드로 돌게 만들어 둔다. 설정은 `algorithm/.env`에 넣는다.

    LLM_PROVIDER=gemini          # gemini | openai | anthropic (없으면 키 보고 자동)
    GOOGLE_API_KEY=...           # gemini 일 때
    OPENAI_API_KEY=...           # openai 일 때
    ANTHROPIC_API_KEY=...        # anthropic 일 때
    LLM_MODEL=gemini-2.5-flash   # 생략하면 제공자별 기본값

특징
----
- **디스크 캐시**: 프롬프트 해시로 저장한다. 같은 프롬프트를 다시 부르면 API를
  쓰지 않는다 (재실행 비용 0).
- **토큰·비용 집계**: 호출마다 usage를 모아 둔다.
- **가짜 모드**: `FakeLLM`으로 API 없이 코드 검증을 먼저 한다.

새 패키지를 깔지 않는다 (requests 만 쓴다). 환경이 원본과 소수점까지 같아야
하므로 requirements-lock.txt를 건드리지 않는 게 중요하다.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent.parent
ENV_PATH = ROOT / "algorithm" / ".env"
CACHE_DIR = ROOT / "algorithm" / "results" / "llm" / "cache"

DEFAULT_MODELS = {
    "gemini": "gemini-2.5-flash",
    "openai": "gpt-4.1-mini",
    "anthropic": "claude-sonnet-5",
}
KEY_NAMES = {
    "gemini": "GOOGLE_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}


class LLMError(Exception):
    pass


class RateLimited(Exception):
    """429. Google이 알려 준 대기 시간을 들고 있다."""

    def __init__(self, seconds: float, message: str = "") -> None:
        super().__init__(message or f"속도 제한, {seconds:.0f}초 대기")
        self.seconds = seconds


def load_env() -> None:
    """algorithm/.env 를 읽어 os.environ에 올린다 (이미 있는 값은 그대로 둔다)."""
    if not ENV_PATH.exists():
        return
    for line in ENV_PATH.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def detect_provider() -> str:
    load_env()
    explicit = (os.environ.get("LLM_PROVIDER") or "").strip().lower()
    if explicit:
        if explicit not in KEY_NAMES:
            raise LLMError(f"모르는 LLM_PROVIDER: {explicit} (가능: {', '.join(KEY_NAMES)})")
        if not os.environ.get(KEY_NAMES[explicit]):
            raise LLMError(f"LLM_PROVIDER={explicit} 인데 {KEY_NAMES[explicit]} 가 없습니다.")
        return explicit
    for provider, key in KEY_NAMES.items():
        if os.environ.get(key):
            return provider
    raise LLMError(
        "LLM 키가 없습니다. algorithm/.env 에 아래 중 하나를 넣어 주세요:\n"
        "  GOOGLE_API_KEY=...  (또는 OPENAI_API_KEY / ANTHROPIC_API_KEY)\n"
        "  LLM_MODEL=...       (생략 가능)"
    )


@dataclass
class Usage:
    calls: int = 0
    cached: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    rate_limited: int = 0

    def add(self, inp: int, out: int) -> None:
        self.calls += 1
        self.input_tokens += inp
        self.output_tokens += out

    def as_dict(self) -> dict:
        return {
            "api_calls": self.calls,
            "cache_hits": self.cached,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "rate_limit_waits": self.rate_limited,
        }


@dataclass
class LLMClient:
    """제공자에 상관없이 `ask(prompt) -> str` 하나만 노출한다."""

    provider: str = ""
    model: str = ""
    cache_dir: Path = CACHE_DIR
    temperature: float = 0.0
    max_output_tokens: int = 1024
    timeout: int = 120
    max_retries: int = 10
    # 무료 등급은 분당 호출 수가 빡빡하다. 호출 사이에 최소 간격을 둬서 429를
    # 애초에 덜 맞는다 (LLM_MIN_INTERVAL 초, 기본 4초 = 분당 15회).
    min_interval: float = 0.0
    usage: Usage = field(default_factory=Usage)

    def __post_init__(self) -> None:
        if not self.provider:
            self.provider = detect_provider()
        load_env()
        if not self.model:
            self.model = os.environ.get("LLM_MODEL") or DEFAULT_MODELS[self.provider]
        self.api_key = os.environ.get(KEY_NAMES[self.provider], "")
        if not self.api_key:
            raise LLMError(f"{KEY_NAMES[self.provider]} 가 없습니다.")
        if not self.min_interval:
            self.min_interval = float(os.environ.get("LLM_MIN_INTERVAL", "4"))
        self._last_call = 0.0
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # -- 캐시 -------------------------------------------------------------
    def _cache_path(self, prompt: str, system: str) -> Path:
        digest = hashlib.sha256(
            f"{self.provider}|{self.model}|{self.temperature}|{system}|{prompt}".encode("utf-8")
        ).hexdigest()[:32]
        return self.cache_dir / f"{digest}.json"

    # -- 호출 -------------------------------------------------------------
    def ask(self, prompt: str, system: str = "") -> str:
        path = self._cache_path(prompt, system)
        if path.exists():
            self.usage.cached += 1
            return json.loads(path.read_text(encoding="utf-8"))["text"]

        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                gap = self.min_interval - (time.time() - self._last_call)
                if gap > 0:
                    time.sleep(gap)
                self._last_call = time.time()
                text, inp, out = self._call(prompt, system)
                self.usage.add(inp, out)
                path.write_text(
                    json.dumps({"provider": self.provider, "model": self.model,
                                "system": system, "prompt": prompt, "text": text,
                                "input_tokens": inp, "output_tokens": out},
                               ensure_ascii=False),
                    encoding="utf-8")
                return text
            except LLMError:
                raise
            except RateLimited as error:
                last_error = error
                self.usage.rate_limited += 1
                wait = max(error.seconds, min(2 ** attempt, 60))
                print(f"      [속도 제한] {wait:.0f}초 쉬고 다시 시도 "
                      f"({attempt + 1}/{self.max_retries})", flush=True)
                time.sleep(wait)
            except Exception as error:  # 네트워크/일시적 오류만 재시도
                last_error = error
                time.sleep(min(2 ** attempt, 60))
        raise LLMError(f"{self.max_retries}회 재시도 후 실패: {last_error}")

    def _call(self, prompt: str, system: str) -> tuple[str, int, int]:
        if self.provider == "gemini":
            return self._call_gemini(prompt, system)
        if self.provider == "openai":
            return self._call_openai(prompt, system)
        return self._call_anthropic(prompt, system)

    def _check(self, response: requests.Response) -> dict:
        if response.status_code == 429:
            seconds = 30.0
            try:
                for detail in response.json().get("error", {}).get("details", []):
                    delay = detail.get("retryDelay")
                    if delay and str(delay).endswith("s"):
                        seconds = float(str(delay)[:-1])
            except Exception:
                pass
            raise RateLimited(seconds)
        if response.status_code == 404:
            raise LLMError(
                f"모델 이름을 찾을 수 없습니다: {self.model} ({self.provider}). "
                f"algorithm/.env 의 LLM_MODEL 을 실제 쓸 수 있는 이름으로 바꿔 주세요.\n"
                f"응답: {response.text[:300]}")
        if response.status_code in (401, 403):
            raise LLMError(f"API 키가 거부되었습니다 ({response.status_code}). "
                           f"응답: {response.text[:300]}")
        if response.status_code >= 400:
            response.raise_for_status()
        return response.json()

    def _call_gemini(self, prompt: str, system: str) -> tuple[str, int, int]:
        url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
               f"{self.model}:generateContent")
        body: dict = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": self.temperature,
                                 "maxOutputTokens": self.max_output_tokens},
        }
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        data = self._check(requests.post(
            url, params={"key": self.api_key},
            headers={"Content-Type": "application/json"},
            json=body, timeout=self.timeout))
        candidates = data.get("candidates") or []
        if not candidates:
            raise LLMError(f"Gemini 응답에 candidates가 없습니다: {str(data)[:300]}")
        parts = candidates[0].get("content", {}).get("parts") or []
        text = "".join(p.get("text", "") for p in parts)
        usage = data.get("usageMetadata", {})
        return text, int(usage.get("promptTokenCount", 0)), int(usage.get("candidatesTokenCount", 0))

    def _call_openai(self, prompt: str, system: str) -> tuple[str, int, int]:
        messages = ([{"role": "system", "content": system}] if system else []) + \
                   [{"role": "user", "content": prompt}]
        data = self._check(requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}",
                     "Content-Type": "application/json"},
            json={"model": self.model, "messages": messages,
                  "temperature": self.temperature,
                  "max_completion_tokens": self.max_output_tokens},
            timeout=self.timeout))
        text = data["choices"][0]["message"]["content"] or ""
        usage = data.get("usage", {})
        return text, int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0))

    def _call_anthropic(self, prompt: str, system: str) -> tuple[str, int, int]:
        body: dict = {"model": self.model, "max_tokens": self.max_output_tokens,
                      "temperature": self.temperature,
                      "messages": [{"role": "user", "content": prompt}]}
        if system:
            body["system"] = system
        data = self._check(requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": self.api_key, "anthropic-version": "2023-06-01",
                     "Content-Type": "application/json"},
            json=body, timeout=self.timeout))
        text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
        usage = data.get("usage", {})
        return text, int(usage.get("input_tokens", 0)), int(usage.get("output_tokens", 0))


class FakeLLM:
    """API 없이 코드를 검증하기 위한 가짜 LLM.

    `responder(prompt, system) -> str` 를 받아 그대로 돌려준다. 실제 호출 전에
    파싱·채점·저장 경로를 전부 이 가짜로 먼저 통과시킨다.
    """

    def __init__(self, responder, model: str = "fake-model") -> None:
        self.responder = responder
        self.provider = "fake"
        self.model = model
        self.usage = Usage()
        self.prompts: list[tuple[str, str]] = []

    def ask(self, prompt: str, system: str = "") -> str:
        self.prompts.append((system, prompt))
        text = self.responder(prompt, system)
        self.usage.add(len(prompt) // 4, len(text) // 4)
        return text


def extract_json(text: str) -> dict:
    """모델이 앞뒤에 군더더기를 붙여도 JSON 한 덩어리를 꺼낸다."""
    text = (text or "").strip()
    if text.startswith("```"):
        lines = [l for l in text.splitlines() if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise LLMError(f"JSON을 찾을 수 없습니다: {text[:200]!r}")
    return json.loads(text[start:end + 1])
