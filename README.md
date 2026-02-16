# 📊 Social Media Content Analyzer

## Overview

The Social Media Content Analyzer is a web-based application that extracts text from PDFs and images and provides engagement optimization insights for social media content.

The application supports both text-based documents and scanned/image-based documents using Optical Character Recognition (OCR).

---

## Features

- Upload PDF or Image files
- Text extraction from:
  - Text-based PDFs (via pdfplumber)
  - Scanned PDFs & Images (via Tesseract OCR)
- Image preprocessing using OpenCV for improved OCR accuracy
- Content metrics:
  - Word count
  - Hashtag count
  - Emoji count
  - Sentence count
- Engagement optimization suggestions
- Downloadable analysis report
- Clean and interactive UI built with Streamlit

---

## Tech Stack

- Python
- Streamlit
- pdfplumber
- pytesseract
- OpenCV
- Pillow

---

## How It Works

1. User uploads a PDF or image file.
2. Text is extracted using pdfplumber or Tesseract OCR depending on file type.
3. Extracted content is preprocessed and analyzed.
4. Engagement metrics and optimization suggestions are generated.

---

## Installation

```bash
pip install -r requirements.txt
streamlit run app.py
