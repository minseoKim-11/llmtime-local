# LLMTime/models/ollama_api.py

import os
import math
import re
import requests
from typing import List, Optional, Any

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://100.106.55.122:6007")

# 숫자 패턴 (부호/정수/소수)
_NUM_RE = re.compile(r"[-+]?(?:\d+(?:\.\d+)?|\.\d+)")


# 설명 모드로 튈 때 자주 나오는 트리거들(안전한 stop들)
# NOTE: ","나 " " 같은 구분자는 stop에 넣으면 첫 구분자에서 끊겨서 망가질 수 있음
DEFAULT_STOP = [
    " is ",
    " are ",
    " because ",
    " therefore ",
    " Wikipedia",
    " wiki",
    "http",
    "https",
    " prime",
    " composite",
    " number",
    " composed",
    "=",
]


def _strip_ollama_prefix(model: str) -> str:
    return model.replace("ollama/", "", 1)


def _get_time_sep(settings: Optional[Any]) -> str:
    return getattr(settings, "time_sep", ",")


def _get_digit_sep(settings: Optional[Any]) -> str:
    """
    LLMTime serialize는 종종 자릿수(또는 bit)를 공백으로 분리:
      668 -> "6 6 8"
    프로젝트마다 필드명이 다를 수 있어 방어적으로 탐색.
    """
    for name in ("bit_sep", "digit_sep", "token_sep", "sep", "value_sep"):
        if settings is not None and hasattr(settings, name):
            v = getattr(settings, name)
            if isinstance(v, str) and v != "":
                return v
    return " "


def _ensure_trailing_sep(prompt: str, time_sep: str) -> str:
    p = (prompt or "").strip()
    return p if p.endswith(time_sep) else (p + time_sep)


def _parse_ollama_numbers(raw_text: str, time_sep: str) -> List[str]:
    """
    1) time_sep로 split
    2) 각 chunk 내부 공백 제거
    3) 숫자만 추출
    """
    nums: List[str] = []
    if not raw_text:
        return nums

    for chunk in raw_text.split(time_sep):
        compact = re.sub(r"\s+", "", chunk)
        if not compact:
            continue
        m = _NUM_RE.search(compact)
        if not m:
            continue
        nums.append(m.group(0))
    return nums


def _to_digit_spaced(tok: str, digit_sep: str) -> str:
    """
    정수(코드)만 digit spacing:
      "668" -> "6 6 8"
    """
    t = (tok or "").strip()
    if t.isdigit():
        return digit_sep.join(list(t))
    if (t.startswith("+") or t.startswith("-")) and t[1:].isdigit():
        return t[0] + digit_sep.join(list(t[1:]))
    return t


def _build_instruction_prefix(time_sep: str) -> str:
    """
    숫자열 continuation에만 집중하도록 강제.
    (Ollama는 system role이 없어서 prompt prefix로 넣는 방식이 가장 확실)
    """
    return (
        "You are a time series forecasting model.\n"
        "Task: continue the given sequence.\n"
        f"Output format: ONLY integer codes separated by '{time_sep}'.\n"
        "Do NOT output any words, explanations, dates, equations, or extra punctuation.\n"
        "Do NOT restate the input. ONLY output the continuation.\n"
        "Do NOT output long runs of 0. Values must be plausible continuations of the input scale.\n"
    )


def _is_bad_numeric_sample(nums: List[str], min_required: int) -> bool:
    """
    temp는 유지하되, 품질이 심각하게 나쁜 샘플(너무 짧음/0폭주)을 걸러 재시도하기 위한 휴리스틱.
    """
    if len(nums) < min_required:
        return True

    vals: List[int] = []
    for t in nums[:min_required]:
        try:
            vals.append(int(float(t)))
        except Exception:
            return True

    # 0 비율이 높으면 bad
    zero_ratio = sum(v == 0 for v in vals) / len(vals)
    if zero_ratio >= 0.6:
        return True

    # 긴 0-run이면 bad
    max_run = 0
    run = 0
    for v in vals:
        if v == 0:
            run += 1
            max_run = max(max_run, run)
        else:
            run = 0
    if max_run >= 10:
        return True

    return False


def ollama_completion_fn(
    model: str,
    input_str: str,
    steps: int,
    num_samples: int,
    temp: float,  # 논문 수치 유지 (호출자가 준 값 그대로)
    settings: Optional[Any] = None,
    max_tokens: int = 512,
    timeout: int = 120,
    **kwargs,
) -> List[str]:
    """
    반환: List[str]
    - deserialize_str가 기대하는 time_sep로 join
    - 정수 코드 토큰은 digit-spacing으로 되돌려 반환(핵심)
    - 나쁜 샘플(짧음/0폭주)은 자동 재시도 (temp는 유지)
    """

    if settings is None:
        settings = kwargs.get("settings", None)

    time_sep = _get_time_sep(settings)
    digit_sep = _get_digit_sep(settings)

    url = f"{OLLAMA_BASE_URL}/api/generate"
    ollama_model = _strip_ollama_prefix(model)

    int_steps = int(float(steps))
    n_samples = int(num_samples)

    # llmtime는 steps*STEP_MULTIPLIER로 요청하므로 넉넉히 반환
    target_len = max(30, int(math.ceil(int_steps * 1.5)))

    # 재시도/필터 파라미터 (기본값 안전하게)
    max_retries = int(kwargs.get("max_retries", 3))
    # 실제로는 horizon(steps)만 확보되면 deserialize에 충분.
    # steps는 llmtime.py에서 steps*STEP_MULTIPLIER로 넘어오므로, 여기서는 int_steps 기준으로 잡되 너무 빡세지 않게.
    min_required = int(kwargs.get("min_required", max(12, min(int_steps, 24))))

    # instruction + prompt
    instr = _build_instruction_prefix(time_sep)
    prompt = instr + _ensure_trailing_sep(input_str, time_sep)

    # 옵션: temp는 반드시 유지
    top_p = float(kwargs.get("top_p", 0.9))
    repeat_penalty = float(kwargs.get("repeat_penalty", 1.1))

    stop = kwargs.get("stop", DEFAULT_STOP)

    results: List[str] = []

    for i in range(n_samples):
        payload = {
            "model": ollama_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": float(temp),       # 유지
                "top_p": top_p,
                "repeat_penalty": repeat_penalty,
                "num_predict": int(max(max_tokens, target_len * 8)),
                "stop": stop,
            },
        }

        best_nums: List[str] = []
        best_score: float = -1.0  # 간단한 품질 점수(높을수록 좋음)

        try:
            for attempt in range(max_retries + 1):
                print(f"   [Ollama API] Sample {i+1}/{n_samples} requesting (attempt {attempt+1}/{max_retries+1})...")
                r = requests.post(url, json=payload, timeout=timeout)
                r.raise_for_status()

                raw_text = r.json().get("response", "") or ""
                print(f'   [DEBUG Raw Response] >>> "{raw_text}"')

                nums = _parse_ollama_numbers(raw_text, time_sep=time_sep)
                print(f"   [Ollama API] Sample {i+1} extracted {len(nums)} values.")

                # 품질 점수: (짧음 패널티) + (0비율 패널티)
                # score는 비교용이므로 단순하게.
                if nums:
                    head = nums[:min_required]
                    zeros = 0
                    for t in head:
                        try:
                            if int(float(t)) == 0:
                                zeros += 1
                        except Exception:
                            zeros += 1
                    zero_ratio = zeros / max(1, len(head))
                else:
                    zero_ratio = 1.0

                score = float(len(nums)) - 50.0 * zero_ratio  # 길게 + 0적게 선호

                if score > best_score:
                    best_score = score
                    best_nums = nums

                # good면 바로 채택하고 종료
                if not _is_bad_numeric_sample(nums, min_required=min_required):
                    best_nums = nums
                    break

            nums = best_nums

            # Length Management: steps로 자르지 않음, 부족하면 마지막 값으로 패딩
            if len(nums) == 0:
                selected = ["0"] * target_len
            elif len(nums) < target_len:
                selected = nums + [nums[-1]] * (target_len - len(nums))
            else:
                selected = nums  # 자르지 않음

            # digit-spacing 복원(핵심)
            selected_fmt = [_to_digit_spaced(t, digit_sep=digit_sep) for t in selected]

            # " <sep> "로 join (deserialize_str는 sep 기준 split이라 공백 있어도 OK)
            results.append(f" {time_sep} ".join(selected_fmt))

        except Exception as e:
            print(f"[ollama_completion_fn] error: {e}")
            fallback = [_to_digit_spaced("0", digit_sep=digit_sep)] * target_len
            results.append(f" {time_sep} ".join(fallback))

    return results


def ollama_nll_fn(model, input_str, target_str, *args, **kwargs):
    """
    인터페이스 mismatch 방지용: 항상 0.0
    """
    return 0.0
