import streamlit as st
import pandas as pd
import re
import base64
import requests
from PIL import Image

# 🔁 중의어 정의
ambiguous_meanings = {
    "기술": {"2": "사물을 잘 다룰 수 있는 방법이나 능력 (기능/방법)", "3": "열거하거나 기록하여 서술함 (기록/서술)"},
    "유형": {"2": "성질이나 특징 따위가 공통적인 것끼리 묶은 하나의 틀", "3": "형상이거나 형체가 있음"},
    "의지": {"2": "이루고자 하는 마음(결심)", "3": "기대다 (의지하다)"},
    "지적": {"2": "지시/지목", "3": "지식이나 지성에 관한 것"}
}
ambiguous_words = set(ambiguous_meanings.keys())

prefixes = ["비", "미", "불", "무", "반", "부", "탈", "비非"]  # 접두사 목록

# 🔁 사고도구어 사전 로딩
@st.cache_data
def load_vocab():
    df = pd.read_csv("사고도구어(1~4등급)(가공).csv", encoding="utf-8-sig")
    vocab = {}
    for col, level in zip(df.columns, range(1, 5)):
        for word in df[col].dropna():
            vocab[word.strip()] = level
    return vocab

# 🔁 온독지수 등급 범위 로딩
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

# 🔁 Vision API OCR 호출
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

# 🔁 온독지수 계산 함수 (kiwi 없이)
def calculate_onread_index(text, vocab_dict, grade_ranges, user_choices=None):
    sorted_vocab = sorted(vocab_dict.items(), key=lambda x: -len(x[0]))
    tokens = re.findall(r"\b[\w가-힣]+\b", text)

    seen, used, total, weighted = set(), [], 0, 0

    for token in tokens:
        # 1. 사용자 선택한 중의어 우선
        if token in ambiguous_words and user_choices and token in user_choices:
            level = user_choices[token]
            if token not in seen:
                seen.add(token)
                used.append((token, level))
                total += 1
                weighted += level
            continue

        # 2. 정확 일치
        matched = False
        for word, level in sorted_vocab:
            if token == word and word not in seen:
                seen.add(word)
                used.append((word, level))
                total += 1
                weighted += level
                matched = True
                break
        if matched:
            continue

        # 3. 접두사 제거 후 어근 매칭
        for prefix in prefixes:
            if token.startswith(prefix):
                stem = token[len(prefix):]
                if stem in vocab_dict and stem not in seen:
                    level = user_choices[stem] if stem in ambiguous_words and user_choices and stem in user_choices else vocab_dict[stem]
                    seen.add(stem)
                    used.append((stem, level))
                    total += 1
                    weighted += level
                    break

    if total == 0:
        return 0, "사고도구어가 감지되지 않았습니다.", [], 0, 0

    word_tokens = re.findall(r"[\w가-힣]+", text)
    cttr = min(len(seen) / (2 * total) ** 0.5, 1.0)
    norm_weight = weighted / (4 * total)
    density = total / len(word_tokens)
    index = ((0.7 * cttr + 0.3 * norm_weight) * 500 + 100) * (0.5 + 0.5 * density)

    if len(word_tokens) < 5:
        index *= 0.6

    matched = [g for s, e, g in grade_ranges if s <= index < e]
    level = "~".join(matched) if matched else "해석 불가"
    return round(index), level, used, total, len(word_tokens)

# ✅ Streamlit 앱 시작
st.title("📘 온독지수 자동 분석기")

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

if trigger or any(st.session_state.get(f"choice_{w}") for w in ambiguous_words):
    input_text = st.session_state.get("manual") if input_method == "문장 입력" else st.session_state.get("ocr_text")
    if input_text:
        user_choices = {}
        for word in ambiguous_words:
            if word in input_text:
                st.markdown(f"🔍 **‘{word}’의 의미를 선택하세요:**")
                options = ambiguous_meanings[word]
                selected = st.radio(
                    f"{word} 의미 선택", 
                    options=[("2", options["2"]), ("3", options["3"])],
                    format_func=lambda x: f"{x[0]}등급: {x[1]}", 
                    key=f"choice_{word}"
                )
                user_choices[word] = int(selected[0])

        score, level, used_words, total_count, total_words = calculate_onread_index(input_text, vocab_dict, grade_ranges, user_choices)

        if score == 0:
            st.warning(level)
        else:
            st.success(f"✅ 온독지수: {score}점 ({level})")
            st.caption(f"총 단어 수: {total_words}, 사고도구어 수: {total_count}")
            if total_count < 3:
                st.info("문장이 짧거나 사고도구어가 적어 분석 결과의 신뢰도가 낮을 수 있습니다.")
            if score > 500:
                st.info("💡 고3 이상 수준입니다.")
            if used_words:
                st.markdown("### 사용된 사고도구어")
                for w, l in used_words:
                    st.markdown(f"- **{w}**: {l}등급")
    else:
        st.warning("❗ 문장을 입력한 뒤 '분석하기' 버튼을 눌러주세요.")
