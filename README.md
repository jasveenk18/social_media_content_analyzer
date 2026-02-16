#  Social Media Content Analyzer

##  Overview

The Social Media Content Analyzer is a web-based application that extracts and analyzes textual content from PDF and image files. It combines traditional PDF parsing with Optical Character Recognition (OCR) to support both text based and scanned documents.

After extracting the content, the system performs rule-based analysis to generate actionable engagement insights tailored for social media platforms. It evaluates word count, hashtag usage, emoji density, sentence structure, and the presence of call to action phrases. Based on these metrics, it provides optimization suggestions to improve readability, engagement, and overall reach.

The application is built using Streamlit for an interactive UI, pdfplumber for PDF parsing, Tesseract OCR for text recognition, and OpenCV for advanced image preprocessing. The system is lightweight, deployable, and designed to simulate a practical content optimization workflow.

---

##  Live Application

🔗 https://socialmediacontentanalyzer-ean2k624aopfwhjrabeyst.streamlit.app/

---

##  Features

-  Text extraction from standard PDFs  
-  OCR-based extraction for scanned PDFs and images  
-  Advanced image preprocessing using OpenCV  
-  Content metrics:
  - Word count  
  - Hashtag count  
  - Emoji count  
  - Sentence count  
-  Engagement optimization suggestions  
-  Auto-generated optimized caption  
-  Downloadable analysis report  
-  Clean and responsive Streamlit interface  

---

##  Tech Stack

- Python 3.x  
- Streamlit  
- pdfplumber  
- pytesseract  
- OpenCV  
- Pillow  
- NumPy  

---

##  How It Works

1. The user uploads a PDF or image file.  
2. Text extraction:
   - `pdfplumber` is used for text-based PDFs.
   - `Tesseract OCR` is used for scanned PDFs and images.
3. Image preprocessing improves OCR accuracy using:
   - Grayscale conversion  
   - Denoising  
   - Adaptive thresholding  
   - Morphological transformations  
4. Extracted text is analyzed using rule-based logic.  
5. Engagement insights and optimization tips are displayed.  

---

##  Content Analysis Logic

The analyzer evaluates:

- Caption length (optimal range)  
- Hashtag density  
- Emoji usage  
- Presence of call to action phrases  
- Engagement triggers such as questions  

Suggestions are generated to enhance social media performance while maintaining clarity and structure.

---

##  Installation (Local Setup)

Clone the repository:

```bash
git clone https://github.com/jasveenk18/social_media_content_analyzer.git
cd social_media_content_analyzer
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

---

##  Deployment Notes (Streamlit Cloud)

Ensure the repository includes:

### requirements.txt

```
streamlit
pdfplumber
pytesseract
pillow
opencv-python-headless
numpy
```

### packages.txt

```
tesseract-ocr
```

---

##  Project Structure

```
social_media_content_analyzer/
│
├── app.py
├── requirements.txt
├── packages.txt
├── README.md
└── .gitignore
```

---

##  Use Cases

- Social media managers  
- Content creators  
- Digital marketing students  
- Technical assessment submissions  
- Caption optimization workflows  

---

##  Developed By

Jasveen Kaur  
Software Engineer Technical Assessment
