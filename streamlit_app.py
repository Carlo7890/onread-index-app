import streamlit as st
import pandas as pd
import requests
import json
import re

# ✅ Gemini API 키
API_KEY = st.secrets["gemini_api_key"]

# ✅ 중복 단어 의미 기준
ambiguous_meanings = {
    "기술": {"2": "기능/방법", "3": "기록/서술"},
    "유형": {"2": "공통적인 것끼리 묶은 틀", "3": "형체가 있는 것"},
    "의지": {"2": "이루고자 하는 마음", "3": "의지하다 (기대다)"},
    "지적": {"2": "지시/지목", "3": "지식이나 지성에 관한 것"}
}

# ✅ 중복 단어 리스트
ambiguous_words = list(ambiguous_meanings.keys())

# ✅ Gemini 문맥 등급 판단 함수
def classify_ambiguous_word(word, sentence):
    meaning = ambiguous_meanings.get(word)
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
    data = {"contents": [{"parts": [{"text": prompt}]}]}

    response = requests.post(f"{url}?key={API_KEY}", headers=headers, data=json.dumps(data))
    try:
        reply = response.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
        if "2" in reply:
            return 2
        elif "3" in reply:
            return 3
    except:
        return None

# ✅ 사고도구어 CSV 불러오기
@st.cache_data
def load_vocab(csv_path="사고도구어(1~4등급)(가공).csv"):
    df = pd.read_csv(csv_path)
    vocab_dict = {}
    for col_idx, level in enumerate([1, 2, 3, 4]):
        col = df.columns[col_idx]
        words = df[col].dropna().astype(str).str.strip()
        for word in words:
            if word in ambiguous_words:
                continue  # 중복 단어는 Gemini로 처리
            vocab_dict[word] = level
    return vocab_dict

# ✅ 등급 범위 파일 불러오기
@st.cache_data
def load_grade_ranges(path="온독지수범위.csv"):
    df = pd.read_csv(path)
    ranges = []
    for _, row in df.iterrows():
        try:
            start, end = map(int, str(row["온독지수 범위"]).split("~"))
            ranges.append((start, end, row["대상 학년"]))
        except:
            continue
    return ranges

# ✅ 온독지수 계산 함수
def calculate_onread_index(text, vocab_dict, grade_ranges):
    tokens = re.findall(r"[\w가-힣]+", text)
    seen, used, total, weighted = set(), [], 0, 0

    for token in tokens:
        base = token.strip()
        if base in ambiguous_words:
            level = classify_ambiguous_word(base, text)
            if level:
                seen.add(base)
                used.append((base, level))
                total += 1
                weighted += level
            continue
        for vocab_word, level in vocab_dict.items():
            if vocab_word in base:
                if base not in seen:
                    seen.add(base)
                    used.append((base, level))
                    total += 1
                    weighted += level
                break

    if total == 0:
        return 0, "사고도구어가 감지되지 않았습니다.", [], 0, 0

    cttr = min(len(seen) / (2 * total)**0.5, 1.0)
    norm_weight = weighted / (4 * total)
    density = total / len(tokens)
    index = ((0.7 * cttr + 0.3 * norm_weight) * 500 + 100) * (0.5 + 0.5 * density)

    matched = [g for s, e, g in grade_ranges if s <= index < e]
    level = "~".join(matched) if matched else "해석 불가"
    return round(index), level, used, total, len(tokens)

# ✅ Streamlit UI 구성
st.title("📘 온독지수 분석기 (Gemini 보조 기반)")
vocab_dict = load_vocab()
grade_ranges = load_grade_ranges()

text = st.text_area("문장을 입력하세요", key="manual")
if st.button("🔍 분석하기"):
    if text.strip():
        score, level, used_words, total_count, total_words = calculate_onread_index(text, vocab_dict, grade_ranges)
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
