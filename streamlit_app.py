import streamlit as st
import pandas as pd
import re
import base64
import requests
from PIL import Image
from kiwipiepy import Kiwi

kiwi = Kiwi()

# ✅ Gemini API Key 불러오기
API_KEY = st.secrets["gemini_api_key"]

# ✅ Gemini 판단 필요한 중복 단어와 의미
ambiguous_meanings = {
    "기술": {"2": "기능이나 방법에 관한 것", "3": "기록이나 서술에 관한 것"},
    "유형": {"2": "공통적인 것끼리 묶은 틀", "3": "형체가 있는 것"},
    "의지": {"2": "이루고자 하는 마음", "3": "기대다의 의미"},
    "지적": {"2": "지시나 지목", "3": "지식이나 지성"}
}
ambiguous_words = list(ambiguous_meanings.keys())

# ✅ Gemini 문맥 분석 함수
def classify_ambiguous_word(word, sentence):
    meaning = ambiguous_meanings[word]
    prompt = f"""
문장: "{sentence}"
단어: "{word}"

다음 중 어떤 의미로 쓰였는가?

- 2등급 의미: {meaning['2']}
- 3등급 의미: {meaning['3']}

문맥에 맞는 등급을 숫자로만 정확하게 답해주세요. (2 또는 3)
"""

    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{"parts": [{"text": prompt}]}]
    }
    response = requests.post(f"{url}?key={API_KEY}", headers=headers, json=payload)
    try:
        reply = response.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
        if "2" in reply:
            return 2
        elif "3" in reply:
            return 3
    except:
        return 2  # 오류 발생 시 기본값

# ✅ 사고도구어 불러오기
@st.cache_data
def load_vocab():
    df = pd.read_csv("사고도구어(1~4등급)(가공).csv", encoding="utf-8-sig")
    vocab_dict = {}
    for col, level in zip(df.columns, range(1, 5)):
        for word in df[col].dropna():
            if word in ambiguous_words:
                continue
            vocab_dict[word.strip()] = level
    return vocab_dict

# ✅ 온독지수 범위 불러오기
@st.cache_data
def load_grade_ranges():
    df = pd.read_csv("온독지수범위.csv", encoding="utf-8-sig")
    ranges = []
    for _, row in df.iterrows():
        try:
            start, end = map(int, str(row["온독지수 범위"]).split("~"))
            ranges.append((start, end, row["대상 학년"]))
        except:
            continue
    return ranges

# ✅ Google Vision API OCR
def call_vision_api(image_bytes):
    api_key = st.secrets["vision_api_key"]
    url = f"https://vision.googleapis.com/v1/images:annotate?key={api_key}"
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")
    response = requests.post(url, json={
        "requests": [{"image": {"content": image_base64}, "features": [{"type": "TEXT_DETECTION"}]}]
    })
    try:
        return response.json()["responses"][0]["fullTextAnnotation"]["text"]
    except:
        return ""

# ✅ 온독지수 계산 함수
def calculate_onread_index(text, vocab_dict, grade_ranges):
    tokens = [t.form for t in kiwi.analyze(text)[0][0]]
    seen, used, total, weighted = set(), [], 0, 0

    for token in tokens:
        base = token.strip()
        if base in ambiguous_words:
            level = classify_ambiguous_word(base, text)
            if base not in seen:
                seen.add(base)
                used.append((base, level))
                total += 1
                weighted += level
            continue

        for vocab_word, level in vocab_dict.items():
            if vocab_word in base and base not in seen:
                seen.add(base)
                used.append((base, level))
                total += 1
                weighted += level
                break

    if total == 0:
        return 0, "사고도구어가 감지되지 않았습니다.", [], 0, 0

    word_tokens = re.findall(r"[\w가-힣]+", text)
    cttr = min(len(seen) / (2 * total)**0.5, 1.0)
    norm_weight = weighted / (4 * total)
    density = total / len(word_tokens)
    index = ((0.7 * cttr + 0.3 * norm_weight) * 500 + 100) * (0.5 + 0.5 * density)

    if len(word_tokens) < 5:
        index *= 0.6

    matched = [g for s, e, g in grade_ranges if s <= index < e]
    level = "~".join(matched) if matched else "해석 불가"
    return round(index), level, used, total, len(word_tokens)

# ✅ Streamlit UI
st.title("📘 온독지수 자동 분석기 (Gemini 보조 기반)")

vocab_dict = load_vocab()
grade_ranges = load_grade_ranges()

input_method = st.radio("입력 방법 선택", ("문장 입력", "이미지 업로드"))
trigger, text = False, ""

if input_method == "문장 입력":
    text = st.text_area("문장을 입력하세요", key="manual")
    if st.button("🔍 분석하기"):
        trigger = True
else:
    uploaded = st.file_uploader("이미지 업로드", type=["png", "jpg", "jpeg", "heic"])
    if uploaded:
        image_bytes = uploaded.read()
        st.image(Image.open(uploaded), caption="업로드된 이미지", use_container_width=True)
        ocr_result = call_vision_api(image_bytes).strip()
        st.session_state["ocr"] = ocr_result
    text = st.text_area("📝 OCR 결과 (수정 가능)", value=st.session_state.get("ocr", ""), key="ocr_text")
    if st.button("🔍 분석하기"):
        trigger = True

if trigger:
    input_text = st.session_state.get("manual") if input_method == "문장 입력" else st.session_state.get("ocr_text")
    if input_text:
        score, level, used_words, total_count, total_words = calculate_onread_index(input_text, vocab_dict, grade_ranges)
        if score == 0:
            st.warning(level)
        else:
            st.success(f"✅ 온독지수: {score}점 ({level})")
            st.caption(f"총 단어 수: {total_words}, 사고도구어 수: {total_count}")
            if total_count < 3:
                st.info("문장이 짧거나 사고도구어가 적어 분석 결과의 신뢰도가 낮을 수 있습니다.")
            if score > 500:
                st.info("💡 고3 이상 수준으로 매우 높은 사고 수준입니다.")
            if used_words:
                st.markdown("### 사용된 사고도구어")
                for w, l in used_words:
                    st.markdown(f"- **{w}**: {l}등급")
    else:
        st.warning("❗ 문장을 입력한 뒤 '분석하기' 버튼을 눌러주세요.")

