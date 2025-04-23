import streamlit as st
import pandas as pd
import re
from PIL import Image
from kiwipiepy import Kiwi
import base64
import requests
import os
import datetime

# 형태소 분석기 초기화
kiwi = Kiwi()

@st.cache_data
def load_vocab():
    vocab_file = "사고도구어(1~4등급)(가공).xlsx"
    sheets = pd.read_excel(vocab_file, sheet_name=None)
    word_dict = {}
    for level, df in sheets.items():
        for word in df["단어족"]:
            base_word = str(word).strip()
            word_dict[base_word] = int(level[0])
            # 조사 및 어미 제거형도 포함
            if base_word.endswith("적"):
                word_dict[base_word + "이다"] = int(level[0])
                word_dict[base_word + "으로"] = int(level[0])
                word_dict[base_word + "인"] = int(level[0])
                word_dict[base_word + "임"] = int(level[0])
    return word_dict

@st.cache_data
def load_grade_ranges():
    df = pd.read_excel("온독지수범위.xlsx")
    ranges = []
    for _, row in df.iterrows():
        start, end = map(int, row["온독지수 범위"].split("~"))
        ranges.append((start, end, row["대상 학년"]))
    return ranges

def call_vision_api(image_bytes):
    api_key = st.secrets["vision_api_key"]
    url = f"https://vision.googleapis.com/v1/images:annotate?key={api_key}"

    image_base64 = base64.b64encode(image_bytes).decode("utf-8")
    request_body = {
        "requests": [
            {
                "image": {"content": image_base64},
                "features": [{"type": "TEXT_DETECTION"}]
            }
        ]
    }

    response = requests.post(url, json=request_body)
    if response.status_code == 200:
        result = response.json()
        try:
            return result["responses"][0]["fullTextAnnotation"]["text"]
        except:
            return ""
    else:
        st.error("Google Vision API 요청 실패: " + response.text)
        return ""

def calculate_onread_index(text, vocab_dict, grade_ranges):
    analyzed = kiwi.analyze(text)
    tokens = [token.lemma for token in analyzed[0][0] if token.tag in ('NNG', 'NNP', 'VV', 'VA', 'MAG', 'MM')]

    token_counts = {}
    total = 0
    weighted_sum = 0
    used_words = []
    seen_words = set()
    counted_tokens = set()

    for token in tokens:
        if token in vocab_dict and token not in counted_tokens:
            level = vocab_dict[token]
            token_counts[level] = token_counts.get(level, 0) + 1
            weighted_sum += level
            total += 1
            used_words.append((token, level))
            seen_words.add(token)
            counted_tokens.add(token)

    if total == 0:
        return 0, "사고도구어가 감지되지 않았습니다.", [], 0, 0

    unique = len(seen_words)
    cttr = unique / (2 * total) ** 0.5
    cttr = min(cttr, 1.0)

    norm_weighted = weighted_sum / (4 * total)
    total_words = len(re.findall(r"[\w가-힣]+", text))
    density = total / total_words if total_words > 0 else 0

    density_factor = 0.5 + 0.5 * density
    index = ((0.7 * cttr + 0.3 * norm_weighted) * 500 + 100) * density_factor

    matched_levels = [grade for start, end, grade in grade_ranges if start <= index < end]
    if not matched_levels:
        level = "해석 불가"
    elif len(matched_levels) == 1:
        level = matched_levels[0]
    else:
        level = f"{matched_levels[0]}~{matched_levels[-1]}"

    return round(index), level, used_words, total, total_words
