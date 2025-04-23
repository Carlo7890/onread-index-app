import streamlit as st
import pandas as pd
import re
from PIL import Image
import pytesseract

# pytesseract 한글 인식 설정 (이미지에서 한글 인식)
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'  # 서버 환경에 맞게 경로 조정 필요

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

def calculate_onread_index(text, vocab_dict, grade_ranges):
    tokens = re.findall(r"[\w가-힣]+", text)
    token_counts = {}
    total = 0
    weighted_sum = 0
    used_words = []

    for token in tokens:
        if token in vocab_dict:
            level = vocab_dict[token]
            token_counts[level] = token_counts.get(level, 0) + 1
            weighted_sum += level
            total += 1
            used_words.append((token, level))

    if total == 0:
        return 0, "사고도구어가 감지되지 않았습니다.", []

    unique = len(set([t for t in tokens if t in vocab_dict]))
    cttr = unique / (2 * total) ** 0.5
    norm_weighted = weighted_sum / (4 * total)
    index = ((0.7 * cttr) + (0.3 * norm_weighted)) * 500 + 100

    level = "해석 불가"
    for start, end, grade in grade_ranges:
        if start <= index < end:
            level = grade
            break

    return round(index), level, used_words

st.title("📘 온독지수 자동 분석기")

vocab_dict = load_vocab()
grade_ranges = load_grade_ranges()

input_method = st.radio("입력 방법을 선택하세요:", ("문장 직접 입력", "이미지 업로드"))
text = ""

if input_method == "문장 직접 입력":
    text = st.text_area("분석할 문장을 입력하세요")
elif input_method == "이미지 업로드":
    uploaded_file = st.file_uploader("문장이 담긴 이미지를 업로드하세요", type=["png", "jpg", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="업로드한 이미지", use_column_width=True)
        text = pytesseract.image_to_string(image, lang="kor")  # 한글 OCR
        text = text.strip()
        st.text_area("📝 인식된 한글 텍스트:", value=text, height=150)

if text:
    score, level, used_words = calculate_onread_index(text, vocab_dict, grade_ranges)
    if score == 0:
        st.warning(level)
    else:
        st.success(f"✅ 온독지수: {score}점 ({level})")
        if used_words:
            st.markdown("### 사용된 사고도구어 목록")
            for word, lvl in used_words:
                st.markdown(f"- **{word}**: {lvl}등급")
