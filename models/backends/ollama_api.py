import os
import re
import requests
from typing import List

# 서버/포트는 환경변수로 관리 (기본값은 네가 쓰던 6007)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://100.106.55.122:6007")

# 숫자 추출 정규식 (소수/정수/부호)
NUM_RE = re.compile(r"[-+]?(?:\d*\.\d+|\d+)")

def _strip_ollama_prefix(model: str) -> str:
    # "ollama/llama2:text" -> "llama2:text"
    return model.replace("ollama/", "", 1)

def _ensure_trailing_comma(prompt: str) -> str:
    p = prompt.strip()
    return p if p.endswith(",") else (p + ",")

def ollama_completion_fn(
    model: str,
    input_str: str,
    steps: int,
    num_samples: int,
    temp: float,
    max_tokens: int = 512,
    timeout: int = 120,
    **kwargs,
) -> List[str]:
    """
    Native Ollama /api/generate (stream=false) completion function.
    Returns: list[str], each is comma-separated numbers (length == steps).
    """
    url = f"{OLLAMA_BASE_URL}/api/generate"
    ollama_model = _strip_ollama_prefix(model)

    int_steps = int(float(steps))
    prompt = _ensure_trailing_comma(input_str)

    results: List[str] = []

    for _ in range(num_samples):
        payload = {
            "model": ollama_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": float(temp),
                "num_predict": int(max_tokens),
            },
        }

        try:
            # [수정] 요청을 보내기 직전에 로그를 찍어야 "아, 지금 일하고 있구나"를 압니다.
            print(f"   [Ollama API] Model: {model} | Sample {i+1}/{num_samples} requesting...")

            r = requests.post(url, json=payload, timeout=timeout)
            r.raise_for_status()
            data = r.json()
            raw_text = data.get("response", "")
          
            nums = NUM_RE.findall(raw_text)

            # [로그 추가] 숫자를 몇 개나 찾았는지 실시간 확인
            print(f"   [Ollama API] Sample {i+1} received. Extracted {len(nums)} values.")

            if len(nums) > 0:
                if len(nums) >= int_steps:
                    selected = nums[:int_steps]
                else:
                    # 부족하면 마지막 값으로 패딩
                    selected = nums + [nums[-1]] * (int_steps - len(nums))
                clean = ",".join(selected)
            else:
                print(f"   [Warning] Sample {i+1} contained no numbers!")
                clean = ",".join(["0.0"] * int_steps)

            results.append(clean)

        except Exception as e:
            print(f"[ollama_completion_fn] error: {e}")
            results.append(",".join(["0.0"] * int_steps))

    return results

def ollama_nll_fn(model: str, input_str: str, target_str: str, **kwargs) -> float:
    """
    Native Ollama generate does not expose token logprobs in the default API.
    LLMTime code should tolerate this for sampling-based evaluation,
    but if something expects NLL, you can return 0.0 as a placeholder.
    """
    return 0.0
