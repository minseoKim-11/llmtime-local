import os
import json
import math
import re
import tiktoken
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
from data.serialize import serialize_arr, SerializerSettings
from jax import grad, vmap

DOTENV_PATH = "/workspace/llmtime/.env"

def _client():
    load_dotenv(dotenv_path=DOTENV_PATH, override=True)
    api_key = os.getenv("OPENAI_API_KEY", "EMPTY")
    base_url = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")

    return OpenAI(api_key=api_key, base_url=base_url)

def _is_finite(x: float) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x)

def _parse_final_json(text: str, steps: int):
    """
    기대 포맷(정확히 1줄):
      FINAL_JSON: {"pred":[...]}
    - 반드시 pred 길이가 steps와 같아야 함.
    """
    # 마지막에 있는 FINAL_JSON 블록을 찾는다(가장 뒤에 나온 걸 쓰기 위해 finditer)
    matches = list(re.finditer(r"FINAL_JSON:\s*(\{.*?\})\s*$", text.strip(), flags=re.DOTALL))
    if not matches:
        raise ValueError("No FINAL_JSON block found")

    obj_str = matches[-1].group(1).strip()
    obj = json.loads(obj_str)

    if "pred" not in obj or not isinstance(obj["pred"], list):
        raise ValueError("FINAL_JSON missing 'pred' list")

    arr = obj["pred"]
    if len(arr) != steps:
        raise ValueError(f"pred length mismatch: {len(arr)} != {steps}")

    out = [float(x) for x in arr]
    if not all(_is_finite(x) for x in out):
        raise ValueError("non-finite value in pred")
    return out

def _make_main_prompt(clean_input: str, int_steps: int) -> str:
    return (
        f"Write exactly one line.\n"
        f"That line MUST start with: OUTPUT_LINE:\n"
        f"After OUTPUT_LINE: write exactly {int_steps} integers separated by commas.\n"
        f"No other text. No reasoning.\n"
        f"Sequence: {clean_input}\n"
    )

def _make_repair_prompt(raw_text: str, int_steps: int) -> str:
    return (
        f"Extract the forecast from the text and output exactly ONE line.\n"
        f"That line MUST be:\n"
        f"OUTPUT_LINE: <{int_steps} integers separated by commas>\n"
        f"No other text.\n"
        f"Text:\n{raw_text}\n"
    )


def tokenize_fn(s: str, model: str):
    try:
        enc = tiktoken.encoding_for_model(model)
    except Exception:
        enc = tiktoken.get_encoding("cl100k_base")
    return enc.encode(s)

def get_allowed_ids(strs, model):
    try:
        enc = tiktoken.encoding_for_model(model)
    except Exception:
        enc = tiktoken.get_encoding("cl100k_base")

    ids = []
    for s in strs:
        ids.extend(enc.encode(s))
    return ids


import re

def gpt_completion_fn(model, input_str, steps, settings, num_samples, temp):
    """
    Regex 기반의 무적(Robust) 파싱 전략을 사용하는 완성형 함수입니다.
    JSON 포맷을 지키지 않더라도 텍스트 속에 숫자만 있다면 낚아챕니다.
    """
    int_steps = int(float(steps))
    # LLMTime의 시리얼라이저 설정을 따르되, 기본값은 공백입니다.
    sep = settings.time_sep if hasattr(settings, 'time_sep') else ' '
    
    # 입력 문자열 정리
    clean_input = input_str.replace(" ", "").strip()
    if clean_input.endswith(","):
        clean_input = clean_input[:-1]

    # 시스템 메시지 및 토큰 상한 (충분히 크게 설정)
    sys_msg = "You are a precise time series forecasting engine."
    max_out_tokens = 2500 

    client = _client()

    def _call(messages, n):
        return client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_out_tokens,
            temperature=temp,
            n=n,
        )

    def _zero_fallback() -> str:
        # 상위에서 split()이 실패하지 않도록 명확한 구분자로 0을 채웁니다.
        return sep.join(["0"] * int_steps)

    def _extract_numbers(text: str, expected_len: int) -> str:
        # 1) OUTPUT_LINE: 이 포함된 줄만 가져오기
        print("[TRACE] _extract_numbers: anchor-only mode")

        m = re.search(r"(?m)^OUTPUT_LINE:\s*(.*)$", text)
        if not m:
            raise ValueError("No OUTPUT_LINE line found")
    
        line = m.group(1).strip()
    
        # 2) 정수만 허용
        nums = re.findall(r"-?\d+", line)
    
        if len(nums) != expected_len:
            raise ValueError(f"OUTPUT_LINE has {len(nums)} nums, expected {expected_len}")
    
        return sep.join(nums)


    # 메인 프롬프트 생성 (생각할 시간을 주되 숫자를 명확히 요구)
    user_msg = _make_main_prompt(clean_input, int_steps)
    print(f"\n[DEBUG] FINAL USER MSG SENT TO SERVER:\n{user_msg}")

    try:
        resp = _call(
            messages=[
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": user_msg},
            ],
            n=num_samples
        )
    except Exception as e:
        print(f"[ERROR] API Call failed: {e}")
        return [_zero_fallback()] * num_samples

    results = []

    for idx, c in enumerate(resp.choices):
        try:
            content = getattr(c.message, "content", "") or ""
            reasoning = getattr(c.message, "reasoning_content", "") or ""
            full_text = (content + "\n" + reasoning).strip()

            print(f"\n[DEBUG] Raw Text[{idx}] Length: {len(full_text)}")
            
            # 1순위: 텍스트 전체에서 숫자 낚시 (가장 확률 높음)
            try:
                pred_str = _extract_numbers(full_text, int_steps)
                print(f"[DEBUG] pred_str(head): {pred_str[:120]}")

                results.append(pred_str)
                print(f"[DEBUG] Choice[{idx}] Success via Regex Extraction")
                continue
            except Exception as re_e:
                print(f"[WARN] Regex extraction failed: {re_e}")

            # 2순위: 리페어 시도 (여기서도 숫자 추출 위주로)
            repair_msg = _make_repair_prompt(full_text, int_steps)
            try:
                r_resp = _call(
                    messages=[
                        {"role": "system", "content": sys_msg},
                        {"role": "user", "content": repair_msg},
                    ],
                    n=1
                )
                r_c = r_resp.choices[0]
                r_text = (getattr(r_c.message, "content", "") or "") + "\n" + (getattr(r_c.message, "reasoning_content", "") or "")
                
                pred_str = _extract_numbers(r_text.strip(), int_steps)
                results.append(pred_str)
                print(f"[DEBUG] Choice[{idx}] Success via Repair Extraction")
                continue
            except Exception as r_e:
                print(f"[ERROR] Repair failed: {r_e}")
                results.append(_zero_fallback())

        except Exception as e:
            print(f"[ERROR] Fatal error in choice[{idx}]: {e}")
            results.append(_zero_fallback())

    # 리스트 길이 보정 (num_samples와 일치)
    while len(results) < num_samples:
        results.append(_zero_fallback())
    results = results[:num_samples]

    return results

def gpt_nll_fn(*args, **kwargs):
    raise NotImplementedError("Disabled for now (migration).")
