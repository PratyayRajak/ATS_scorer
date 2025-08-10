import streamlit as st
import pdfplumber
import docx
import pytesseract
from PIL import Image
import io
import re

# ------------------ Helper Functions ------------------ #

def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

def extract_text_from_docx(file):
    doc = docx.Document(file)
    return "\n".join([para.text for para in doc.paragraphs])

def extract_text_from_image(file):
    image = Image.open(file)
    return pytesseract.image_to_string(image)

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def calculate_match_score(resume_text, jd_text):
    resume_words = set(resume_text.lower().split())
    jd_words = set(jd_text.lower().split())
    if not jd_words:
        return 0
    match_count = len(resume_words & jd_words)
    return round((match_count / len(jd_words)) * 100, 2)

# ------------------ Streamlit UI ------------------ #

st.set_page_config(page_title="ATS Resume Scanner", page_icon="🤖", layout="wide")

st.title("🤖 ATS Resume Scanner & Chatbot")

# Sidebar for job description
with st.sidebar:
    st.header("📄 Job Description")
    job_description = st.text_area("Paste the Job Description here", height=300)
    st.markdown("---")
    st.caption("Upload resumes in PDF, DOCX, or image format to analyze.")

# File upload for resumes
uploaded_files = st.file_uploader("Upload Resume(s)", type=["pdf", "docx", "png", "jpg", "jpeg"], accept_multiple_files=True)

if uploaded_files and job_description:
    results = []
    for file in uploaded_files:
        file_type = file.name.split(".")[-1].lower()

        if file_type == "pdf":
            resume_text = extract_text_from_pdf(file)
        elif file_type == "docx":
            resume_text = extract_text_from_docx(file)
        elif file_type in ["png", "jpg", "jpeg"]:
            resume_text = extract_text_from_image(file)
        else:
            st.error(f"Unsupported file format: {file_type}")
            continue

        resume_text = clean_text(resume_text)
        jd_text = clean_text(job_description)
        score = calculate_match_score(resume_text, jd_text)

        results.append({"name": file.name, "score": score, "text": resume_text})

    # Display results
    st.subheader("📊 ATS Results")
    for res in results:
        with st.expander(f"{res['name']} - Match: {res['score']}%"):
            st.write(res["text"])

# ------------------ Chatbot Section ------------------ #

st.markdown("---")
st.subheader("💬 ATS Chatbot Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []

user_input = st.text_input("Type your question about resumes or job description...")

if st.button("Send"):
    if user_input:
        st.session_state.messages.append(("You", user_input))

        if "best resume" in user_input.lower():
            if uploaded_files and job_description:
                best = max(results, key=lambda x: x["score"], default=None)
                if best:
                    bot_reply = f"📌 The best matching resume is **{best['name']}** with a score of {best['score']}%."
                else:
                    bot_reply = "No resumes were analyzed yet."
            else:
                bot_reply = "Please upload resumes and provide a job description first."
        elif "improve" in user_input.lower():
            bot_reply = "✅ Try adding more keywords from the job description into your resume naturally."
        else:
            bot_reply = "I can help with resume ranking, best match, and improvement tips."

        st.session_state.messages.append(("Bot", bot_reply))

# Display chat messages
for sender, msg in st.session_state.messages:
    if sender == "You":
        st.markdown(f"**🧑‍💼 You:** {msg}")
    else:
        st.markdown(f"**🤖 Bot:** {msg}")
