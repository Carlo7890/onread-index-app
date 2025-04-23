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
            word_dict[str(word).strip()] = int(level[0])
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

# ✅ API 호출 횟수 제한 로직 추가
if "daily_calls" not in st.session_state:
    st.session_state["daily_calls"] = 0
    st.session_state["last_reset"] = datetime.date.today()

if st.session_state["last_reset"] != datetime.date.today():
    st.session_state["daily_calls"] = 0
    st.session_state["last_reset"] = datetime.date.today()

if st.session_state["daily_calls"] >= 20:
    st.error("오늘의 할당량을 모두 이용하였습니다. 내일 다시 사용해주세요.")
    st.stop()

st.title("📘 온독지수 자동 분석기")

vocab_dict = load_vocab()
grade_ranges = load_grade_ranges()

input_method = st.radio("입력 방법을 선택하세요:", ("문장 직접 입력", "이미지 업로드"))
text = ""
trigger = False

if input_method == "문장 직접 입력":
    text = st.text_area("분석할 문장을 입력하세요", key="manual_text")
    if st.button("🔍 분석하기"):
        trigger = True
elif input_method == "이미지 업로드":
    uploaded_file = st.file_uploader("문장이 담긴 이미지를 업로드하세요", type=["png", "jpg", "jpeg"])
    ocr_text = ""
    if uploaded_file:
        try:
            image_bytes = uploaded_file.read()
            image = Image.open(uploaded_file)
            st.image(image, caption="업로드한 이미지", use_container_width=True)
            ocr_text = call_vision_api(image_bytes).strip()
            st.session_state["daily_calls"] += 1  # API 호출 시 카운트 증가
            st.session_state["ocr_text"] = ocr_text  # OCR 결과 저장
        except Exception as e:
            st.error(f"이미지를 처리하는 도중 오류가 발생했습니다: {e}")
    text = st.text_area("📝 인식된 한글 텍스트 (수정 가능):", value=st.session_state.get("ocr_text", ""), key="ocr_text_area", height=150)
    if st.button("🔍 분석하기"):
        trigger = True

if trigger:
    current_text = st.session_state.get("manual_text") if input_method == "문장 직접 입력" else st.session_state.get("ocr_text_area")
    if current_text:
        score, level, used_words, total_count, total_words = calculate_onread_index(current_text, vocab_dict, grade_ranges)
        if score == 0:
            st.warning(level)
        else:
            st.success(f"✅ 온독지수: {score}점 ({level})")
            st.caption(f"(총 단어 수: {total_words} / 사고도구어 수: {total_count})")
            if total_count < 3:
                st.info("ℹ️ 문장이 짧아 사고도구어 수가 적지만, 결과는 참고용으로 제공됩니다.")
            if score > 500:
                st.info("💡 온독지수가 고3 수준(500점)을 초과하였습니다. 매우 높은 수준의 사고도구어를 활용하고 있습니다.")
            if used_words:
                st.markdown("### 사용된 사고도구어 목록")
                for word, lvl in used_words:
                    st.markdown(f"- **{word}**: {lvl}등급")
    else:
        st.warning("❗ 문장을 입력한 뒤 분석 버튼을 눌러주세요.")

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
    tokens = [token.form for token in analyzed[0][0] if token.tag in ('NNG', 'NNP', 'VV', 'VA', 'MAG', 'MM')]

    token_counts = {}
    total = 0
    weighted_sum = 0
    used_words = []
    seen_words = set()

    for token in tokens:
        if token in vocab_dict:
            level = vocab_dict[token]
            token_counts[level] = token_counts.get(level, 0) + 1
            weighted_sum += level
            total += 1
            if token not in seen_words:
                used_words.append((token, level))
                seen_words.add(token)

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

# ✅ API 호출 횟수 제한 로직 추가
if "daily_calls" not in st.session_state:
    st.session_state["daily_calls"] = 0
    st.session_state["last_reset"] = datetime.date.today()

if st.session_state["last_reset"] != datetime.date.today():
    st.session_state["daily_calls"] = 0
    st.session_state["last_reset"] = datetime.date.today()

if st.session_state["daily_calls"] >= 20:
    st.error("오늘의 할당량을 모두 이용하였습니다. 내일 다시 사용해주세요.")
    st.stop()

st.title("📘 온독지수 자동 분석기")

vocab_dict = load_vocab()
grade_ranges = load_grade_ranges()

input_method = st.radio("입력 방법을 선택하세요:", ("문장 직접 입력", "이미지 업로드"))
text = ""
trigger = False

if input_method == "문장 직접 입력":
    text = st.text_area("분석할 문장을 입력하세요")
    if st.button("🔍 분석하기"):
        trigger = True
elif input_method == "이미지 업로드":
    uploaded_file = st.file_uploader("문장이 담긴 이미지를 업로드하세요", type=["png", "jpg", "jpeg"])
    ocr_text = ""
    if uploaded_file:
        try:
            image_bytes = uploaded_file.read()
            image = Image.open(uploaded_file)
            st.image(image, caption="업로드한 이미지", use_container_width=True)
            ocr_text = call_vision_api(image_bytes).strip()
            st.session_state["daily_calls"] += 1  # API 호출 시 카운트 증가
        except Exception as e:
            st.error(f"이미지를 처리하는 도중 오류가 발생했습니다: {e}")
    text = st.text_area("📝 인식된 한글 텍스트 (수정 가능):", value=ocr_text, height=150)
    if st.button("🔍 분석하기"):
        trigger = True

if trigger and text:
    score, level, used_words, total_count, total_words = calculate_onread_index(text, vocab_dict, grade_ranges)
    if score == 0:
        st.warning(level)
    else:
        st.success(f"✅ 온독지수: {score}점 ({level})")
        st.caption(f"(총 단어 수: {total_words} / 사고도구어 수: {total_count})")
        if total_count < 3:
            st.info("ℹ️ 문장이 짧아 사고도구어 수가 적지만, 결과는 참고용으로 제공됩니다.")
        if score > 500:
            st.info("💡 온독지수가 고3 수준(500점)을 초과하였습니다. 매우 높은 수준의 사고도구어를 활용하고 있습니다.")
        if used_words:
            st.markdown("### 사용된 사고도구어 목록")
            for word, lvl in used_words:
                st.markdown(f"- **{word}**: {lvl}등급")

