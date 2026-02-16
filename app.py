import streamlit as st
import pdfplumber
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import re
import numpy as np
import cv2

# Uncomment ONLY for local Windows use
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ==========================
# PAGE CONFIGURATION
# ==========================
st.set_page_config(
    page_title="Social Media Content Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================
# CUSTOM STYLING
# ==========================
st.markdown("""
    <style>
        .title { font-size: 42px; font-weight: 700; color: #4CAF50; }
        .subtitle { color: #888; margin-bottom: 30px; font-size: 18px; }
        .card {
            background-color: #1e1e1e;
            padding: 20px;
            border-radius: 12px;
            margin-bottom: 15px;
            border-left: 5px solid #4CAF50;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .metric-box {
            background-color: #262730;
            padding: 20px;
            border-radius: 12px;
            text-align: center;
            border: 1px solid #333;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">📊 Social Media Content Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload PDFs or images → Extract text → Get data-driven engagement insights.</div>', unsafe_allow_html=True)

# ==========================
# FILE UPLOADER
# ==========================
uploaded_file = st.file_uploader(
    "Upload PDF or Image (PDF, PNG, JPG, JPEG)",
    type=["pdf", "png", "jpg", "jpeg"],
    help="Supports text-based PDFs and scanned documents via OCR."
)

# ==========================
# OCR PREPROCESSING
# ==========================
def preprocess_for_ocr(image: Image.Image) -> Image.Image:
    try:
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

        denoised = cv2.fastNlMeansDenoising(gray, h=10)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)

        thresh = cv2.adaptiveThreshold(
            enhanced, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11, 2
        )

        kernel = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        cleaned = cv2.dilate(cleaned, kernel, iterations=1)

        h, w = cleaned.shape
        scale = 2.0 if max(h, w) < 1200 else 1.5
        resized = cv2.resize(cleaned, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

        return Image.fromarray(resized)

    except Exception:
        gray = image.convert("L")
        enhancer = ImageEnhance.Contrast(gray)
        gray = enhancer.enhance(3.0)
        sharpened = gray.filter(ImageFilter.UnsharpMask(radius=2, percent=150))
        return sharpened


# ==========================
# TEXT EXTRACTION
# ==========================
def extract_text_from_pdf(file) -> str:
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            if page_text.strip():
                text += page_text + "\n\n"
            else:
                img = page.to_image(resolution=300).original
                processed = preprocess_for_ocr(img)
                ocr_text = pytesseract.image_to_string(processed, config='--psm 6 --oem 3')
                text += ocr_text + "\n\n"
    return text.strip()


def extract_text_from_image(file) -> str:
    image = Image.open(file)
    processed = preprocess_for_ocr(image)
    return pytesseract.image_to_string(processed, config='--psm 6 --oem 3').strip()


def extract_text(file) -> str:
    try:
        if file.type == "application/pdf":
            return extract_text_from_pdf(file)
        else:
            return extract_text_from_image(file)
    except Exception as e:
        st.error(f"Extraction failed: {str(e)}")
        return ""


# ==========================
# CONTENT ANALYSIS
# ==========================
def analyze_text(text: str):
    if not text.strip():
        return 0, 0, 0, 0, ["No readable text found. Please upload a clearer file."], []

    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()
    word_count = len(words)

    hashtags = re.findall(r'#(\w+)', text)
    hashtag_count = len(hashtags)

    emoji_pattern = re.compile(
        "[\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F1E0-\U0001F1FF]+",
        flags=re.UNICODE
    )
    emoji_count = len(emoji_pattern.findall(text))

    sentence_count = len(re.findall(r'[.!?]+', text))

    suggestions = []

    if word_count > 180:
        suggestions.append("Shorten the caption to improve mobile readability (ideal: under 150 words).")
    elif word_count < 40:
        suggestions.append("Consider adding more depth to increase value and retention.")

    if hashtag_count < 4:
        suggestions.append("Add 4–8 targeted hashtags to improve discoverability.")
    elif hashtag_count > 15:
        suggestions.append("Reduce hashtags to avoid being flagged as spam.")

    if emoji_count < 2:
        suggestions.append("Include 2–4 relevant emojis to enhance engagement.")

    cta_keywords = ['comment', 'like', 'share', 'follow', 'tag', 'dm', 'reply']
    if not any(kw in text.lower() for kw in cta_keywords):
        suggestions.append("Add a call-to-action such as 'Comment below' or 'Share your thoughts'.")

    if not re.search(r'\?', text):
        suggestions.append("Pose a question to encourage interaction.")

    if not suggestions:
        suggestions.append("Content structure is strong. Minor refinements can further optimize performance.")

    return word_count, hashtag_count, emoji_count, sentence_count, suggestions, hashtags


# ==========================
# MAIN APP
# ==========================
if uploaded_file:

    with st.spinner("Processing file..."):
        text = extract_text(uploaded_file)

    if text:
        st.success("Text extracted successfully.")

        col1, col2 = st.columns([1, 1.8])
        with col1:
            if uploaded_file.type != "application/pdf":
                st.image(uploaded_file, caption="Uploaded Preview")
            else:
                st.info("PDF processed successfully.")
        with col2:
            st.subheader("Extracted Content")
            st.text_area("", text, height=280)

        word_count, hashtag_count, emoji_count, sentence_count, suggestions, hashtags = analyze_text(text)

        st.subheader("Content Metrics")
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)

        col_m1.metric("Words", word_count)
        col_m2.metric("Hashtags", hashtag_count)
        col_m3.metric("Emojis", emoji_count)
        col_m4.metric("Sentences", sentence_count)

        if hashtags:
            st.subheader("Detected Hashtags")
            st.code(" ".join([f"#{h}" for h in hashtags]))

        st.subheader("Engagement Optimization Tips")
        for suggestion in suggestions:
            st.markdown(f'<div class="card">• {suggestion}</div>', unsafe_allow_html=True)

        optimized_caption = (
            text[:400] +
            "\n\nWhat are your thoughts? Share in the comments! 👇 #SocialMediaTips"
        )

        st.subheader("Suggested Optimized Caption")
        st.text_area("Copy & paste version:", optimized_caption, height=120)

        st.download_button(
            "Download Analysis as TXT",
            f"EXTRACTED TEXT:\n{text}\n\n"
            f"METRICS:\nWords: {word_count}\nHashtags: {hashtag_count}\n"
            f"Emojis: {emoji_count}\nSentences: {sentence_count}\n\n"
            f"SUGGESTIONS:\n" + "\n".join(suggestions),
            file_name="social_media_analysis.txt"
        )
    else:
        st.warning("No text detected. Please try another file.")

# ==========================
# SIDEBAR
# ==========================
with st.sidebar:
    st.header("About This Project")

    st.markdown("""
    **Technical Assessment Submission**

    - Handles text-based and scanned PDFs  
    - Advanced OCR preprocessing with OpenCV  
    - Content metrics & engagement recommendations  
    - Loading states and structured error handling  

    **Tech Stack**
    - Streamlit
    - pdfplumber
    - pytesseract
    - OpenCV
    - Pillow

    **Deployment Notes**
    Add the following to `requirements.txt`:
    ```
    streamlit
    pdfplumber
    pytesseract
    pillow
    opencv-python-headless
    numpy
    ```

    For Streamlit Cloud, also add `tesseract-ocr` to `packages.txt`.
    """)

st.divider()
st.caption(" ❤️Developed for Software Engineer Technical Assessment")
