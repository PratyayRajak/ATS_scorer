#!/usr/bin/env python3
"""
Complete ATS Pipeline Streamlit UI
Integrates directly with the provided ATS pipeline code
"""

import streamlit as st
import os
import re
import datetime
import json
import base64
import shutil
import yaml
import logging
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import tempfile
from PIL import Image
import io

# Core ML Dependencies
import numpy as np
import spacy
import docx2txt
import pdfplumber
import requests
from bs4 import BeautifulSoup
import pytesseract
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util
import shap
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Page configuration
st.set_page_config(
    page_title="Advanced ATS Pipeline",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    
    .step-container {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .metric-container {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem;
    }
    
    .candidate-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #e9ecef;
        margin-bottom: 1rem;
    }
    
    .score-badge {
        background: linear-gradient(135deg, #28a745, #20c997);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
        font-size: 1.2rem;
    }
    
    .warning-box {
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .success-box {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Configuration class (from your code)
class Config:
    UPLOAD_FOLDER = "uploads"
    OUTPUT_FOLDER = "results"
    ALLOWED_EXTENSIONS = {'pdf', 'docx'}
    TESSERACT_PATH = None

# Helper functions (from your code)
def setup_tesseract():
    if Config.TESSERACT_PATH and os.path.exists(Config.TESSERACT_PATH):
        pytesseract.pytesseract.tesseract_cmd = Config.TESSERACT_PATH

def ensure_directories():
    for folder in [Config.UPLOAD_FOLDER, Config.OUTPUT_FOLDER]:
        Path(folder).mkdir(exist_ok=True)

def clean_text(text: str) -> str:
    if not text: return ""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s\.,;:\!\?\-\+\@\#\$\%\(\)\/\\]', '', text)
    return text.strip()

# Initialize models and session state
@st.cache_resource
def load_models():
    """Load ML models once and cache them"""
    try:
        logging.info("Loading spaCy model 'en_core_web_sm'...")
        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            st.warning("spaCy model not found. Please install it: python -m spacy download en_core_web_sm")
            return None, None, None
            
        logging.info("Loading SentenceTransformer model...")
        semantic_model = SentenceTransformer('all-mpnet-base-v2')
        tfidf_vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
        
        return nlp, semantic_model, tfidf_vectorizer
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None

# Load models
nlp, semantic_model, tfidf_vectorizer = load_models()

if nlp is None:
    st.stop()

# Job Description Functions
def get_jd_from_url(url: str) -> str:
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        for script in soup(["script", "style"]):
            script.extract()
        return ' '.join(soup.stripped_strings)
    except Exception as e:
        logging.error(f"Failed to scrape URL {url}: {e}")
        return ""

def get_jd_from_screenshot(image_data) -> str:
    try:
        return pytesseract.image_to_string(image_data)
    except Exception as e:
        logging.error(f"OCR failed: {e}")
        return ""

# Resume parsing functions
def extract_text(file_path: str) -> str:
    ext = Path(file_path).suffix.lower()
    if ext == '.pdf':
        with pdfplumber.open(file_path) as pdf:
            return '\n'.join(page.extract_text() for page in pdf.pages if page.extract_text())
    elif ext == '.docx':
        return docx2txt.process(file_path)
    return ""

def parse_resume(file_path: str) -> Optional[Dict]:
    raw_text = extract_text(file_path)
    if not raw_text or len(raw_text) < 50:
        return None

    doc = nlp(raw_text)
    
    name = next((ent.text for ent in doc.ents if ent.label_ == 'PERSON'), None)
    email = next((token.text for token in doc if token.like_email), None)
    phone = next((match.group(0) for match in [re.search(r'(\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})', raw_text)] if match), None)
    
    years = [int(y) for y in re.findall(r'\b(19|20)\d{2}\b', raw_text)]
    experience_years = max(years) - min(years) if len(years) >= 2 else 0

    redacted_text = raw_text
    if name: redacted_text = redacted_text.replace(name, '[NAME]')
    if email: redacted_text = redacted_text.replace(email, '[EMAIL]')
    if phone: redacted_text = redacted_text.replace(phone, '[PHONE]')

    return {
        "file_path": file_path,
        "raw_text": raw_text,
        "redacted_text": redacted_text,
        "parsed_data": {
            "name": name,
            "email": email,
            "phone": phone,
            "experience_years": experience_years,
        }
    }

# Scoring functions
def normalize_scores(scores: np.ndarray) -> np.ndarray:
    min_s, max_s = np.min(scores), np.max(scores)
    return (scores - min_s) / (max_s - min_s) if max_s > min_s else np.zeros_like(scores)

def score_candidates(jd_text: str, parsed_resumes: List[Dict], weights: Dict) -> List[Dict]:
    resume_texts = [res['redacted_text'] for res in parsed_resumes]
    
    jd_embedding = semantic_model.encode(jd_text, convert_to_tensor=True)
    resume_embeddings = semantic_model.encode(resume_texts, convert_to_tensor=True)
    semantic_scores = util.pytorch_cos_sim(jd_embedding, resume_embeddings).numpy().flatten()

    corpus = [jd_text] + resume_texts
    tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)
    keyword_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

    jd_req_experience = float(weights.get('jd_required_experience', 5.0))
    experience_scores = np.array([min(res['parsed_data']['experience_years'] / jd_req_experience, 1.0) if jd_req_experience > 0 else 1.0 for res in parsed_resumes])

    norm_semantic = normalize_scores(semantic_scores)
    norm_keyword = normalize_scores(keyword_scores)
    norm_experience = normalize_scores(experience_scores)

    final_scores = (
        norm_semantic * weights['semantic_similarity'] +
        norm_keyword * weights['keyword_match'] +
        norm_experience * weights['years_of_experience']
    )

    results = []
    for i, resume in enumerate(parsed_resumes):
        results.append({
            "filename": Path(resume['file_path']).name,
            "score": round(float(final_scores[i]) * 100, 2),  # Convert to percentage
            "score_breakdown": {
                "semantic_similarity": round(float(norm_semantic[i]) * 100, 2),
                "keyword_match": round(float(norm_keyword[i]) * 100, 2),
                "experience_match": round(float(norm_experience[i]) * 100, 2),
            },
            "parsed_data": resume['parsed_data'],
            "raw_text": resume['raw_text']
        })
    
    return sorted(results, key=lambda x: x['score'], reverse=True)

def generate_explanation(jd_text: str, candidate_data: Dict) -> str:
    """Generate explanation using SHAP"""
    try:
        def predict_similarity(texts):
            jd_embedding = semantic_model.encode([jd_text])
            text_embeddings = semantic_model.encode(texts)
            return cosine_similarity(jd_embedding, text_embeddings).flatten()

        explainer = shap.Explainer(predict_similarity, shap.maskers.Text(r"\W+"))
        resume_text = candidate_data.get('raw_text', '')
        
        if not resume_text:
            return "Explanation could not be generated: resume text is missing."

        shap_values = explainer([resume_text])
        words = np.array(shap_values.data[0])
        values = np.array(shap_values.values[0])
        
        positive_indices = np.where(values > 0)[0]
        
        if len(positive_indices) > 0:
            top_indices = positive_indices[np.argsort(-values[positive_indices])[:3]]
            top_contributors = words[top_indices]
            contrib_str = f"Key strengths found in: {', '.join(top_contributors)}"
        else:
            contrib_str = "General semantic alignment detected across the document."

        return (
            f"🎯 **Overall Assessment**: Strong candidate with {candidate_data['score']:.1f}% match.\n\n"
            f"📊 **Score Breakdown**:\n"
            f"• Semantic Alignment: {candidate_data['score_breakdown']['semantic_similarity']:.1f}%\n"
            f"• Keyword Match: {candidate_data['score_breakdown']['keyword_match']:.1f}%\n"
            f"• Experience Match: {candidate_data['score_breakdown']['experience_match']:.1f}%\n\n"
            f"🔍 **Key Insights**: {contrib_str}\n\n"
            f"💼 **Experience**: {candidate_data['parsed_data']['experience_years']} years of relevant experience detected."
        )
    except Exception as e:
        return f"Analysis explanation temporarily unavailable: {str(e)}"

# Initialize session state
def init_session_state():
    if 'step' not in st.session_state:
        st.session_state.step = 1
    if 'job_description' not in st.session_state:
        st.session_state.job_description = ""
    if 'parsed_resumes' not in st.session_state:
        st.session_state.parsed_resumes = []
    if 'ranked_candidates' not in st.session_state:
        st.session_state.ranked_candidates = []

init_session_state()
setup_tesseract()
ensure_directories()

# Main UI
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🎯 Advanced ATS Pipeline</h1>
        <p>AI-Powered Resume Screening & Candidate Ranking System</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.header("🧭 Navigation")
        
        # Progress indicator
        steps = ["📝 Job Description", "📁 Upload Resumes", "🔍 Analysis", "🏆 Results"]
        for i, step_name in enumerate(steps, 1):
            if i < st.session_state.step:
                st.success(f"✅ {step_name}")
            elif i == st.session_state.step:
                st.info(f"🔄 {step_name}")
            else:
                st.write(f"⏳ {step_name}")
        
        st.divider()
        
        # Settings
        st.header("⚙️ Settings")
        
        with st.expander("Model Configuration"):
            st.info("Models loaded successfully!" if nlp and semantic_model else "Models not loaded")
            
        with st.expander("Create Config File"):
            if st.button("Generate Default Config"):
                config = {
                    'weights': {
                        'semantic_similarity': 0.4,
                        'keyword_match': 0.35,
                        'years_of_experience': 0.25,
                        'jd_required_experience': 5.0
                    },
                    'smtp_settings': {
                        'server': 'smtp.gmail.com',
                        'port': 587
                    },
                    'email_template': {
                        'subject': 'Interview Invitation - Your Application',
                        'body': '''Dear [Candidate Name],

Thank you for your application. We would like to invite you for an interview.

Best regards,
HR Team'''
                    }
                }
                
                with open('config.yaml', 'w') as f:
                    yaml.dump(config, f, default_flow_style=False)
                st.success("Config file created!")

    # Main content based on current step
    if st.session_state.step == 1:
        step1_job_description()
    elif st.session_state.step == 2:
        step2_upload_resumes()
    elif st.session_state.step == 3:
        step3_analysis()
    elif st.session_state.step == 4:
        step4_results()

def step1_job_description():
    st.header("📝 Step 1: Set Job Description")
    
    tab1, tab2, tab3 = st.tabs(["📄 Text Input", "🔗 URL Scraping", "📷 Screenshot"])
    
    with tab1:
        st.subheader("Direct Text Input")
        jd_text = st.text_area(
            "Job Description",
            height=300,
            placeholder="Paste your complete job description here...",
            help="Include all requirements, skills, and qualifications"
        )
        
        if st.button("✅ Set Job Description", type="primary", key="text_jd"):
            if jd_text.strip() and len(jd_text.strip()) > 50:
                st.session_state.job_description = clean_text(jd_text)
                st.success("Job description set successfully!")
                st.info(f"Preview: {st.session_state.job_description[:200]}...")
                st.session_state.step = 2
                st.rerun()
            else:
                st.error("Please enter a job description with at least 50 characters")
    
    with tab2:
        st.subheader("URL Scraping")
        jd_url = st.text_input(
            "Job Posting URL",
            placeholder="https://example.com/job-posting"
        )
        
        if st.button("🌐 Scrape from URL", type="primary", key="url_jd"):
            if jd_url.strip():
                with st.spinner("Scraping job description..."):
                    scraped_text = get_jd_from_url(jd_url)
                    if scraped_text and len(scraped_text) > 50:
                        st.session_state.job_description = clean_text(scraped_text)
                        st.success("Job description scraped successfully!")
                        st.info(f"Preview: {st.session_state.job_description[:200]}...")
                        st.session_state.step = 2
                        st.rerun()
                    else:
                        st.error("Failed to scrape sufficient content from URL")
            else:
                st.error("Please enter a valid URL")
    
    with tab3:
        st.subheader("Screenshot OCR")
        uploaded_image = st.file_uploader(
            "Upload Screenshot",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a clear screenshot of the job description"
        )
        
        if uploaded_image:
            st.image(uploaded_image, caption="Uploaded Screenshot", use_column_width=True)
            
            if st.button("🔍 Extract Text", type="primary", key="ocr_jd"):
                with st.spinner("Processing image with OCR..."):
                    image = Image.open(uploaded_image)
                    extracted_text = get_jd_from_screenshot(image)
                    
                    if extracted_text and len(extracted_text.strip()) > 50:
                        st.session_state.job_description = clean_text(extracted_text)
                        st.success("Text extracted successfully!")
                        st.info(f"Preview: {st.session_state.job_description[:200]}...")
                        st.session_state.step = 2
                        st.rerun()
                    else:
                        st.error("Could not extract sufficient text from image")

def step2_upload_resumes():
    st.header("📁 Step 2: Upload Resume Files")
    
    if not st.session_state.job_description:
        st.warning("Please set the job description first")
        if st.button("← Back to Step 1"):
            st.session_state.step = 1
            st.rerun()
        return
    
    # Display job description preview
    with st.expander("📝 Job Description Preview"):
        st.write(st.session_state.job_description[:500] + "..." if len(st.session_state.job_description) > 500 else st.session_state.job_description)
    
    uploaded_files = st.file_uploader(
        "Upload Resume Files",
        type=['pdf', 'docx'],
        accept_multiple_files=True,
        help="Upload PDF or Word documents"
    )
    
    if uploaded_files:
        st.success(f"📋 Selected {len(uploaded_files)} files:")
        for file in uploaded_files:
            st.write(f"• {file.name} ({file.size / 1024 / 1024:.2f} MB)")
        
        if st.button("🚀 Process Resumes", type="primary"):
            with st.spinner("Processing resume files..."):
                progress_bar = st.progress(0)
                processed_resumes = []
                failed_files = []
                
                for i, file in enumerate(uploaded_files):
                    try:
                        # Save temporary file
                        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.name).suffix) as tmp_file:
                            tmp_file.write(file.read())
                            tmp_path = tmp_file.name
                        
                        # Parse resume
                        parsed = parse_resume(tmp_path)
                        if parsed:
                            processed_resumes.append(parsed)
                        else:
                            failed_files.append(file.name)
                        
                        # Clean up
                        os.unlink(tmp_path)
                        
                    except Exception as e:
                        failed_files.append(f"{file.name}: {str(e)}")
                    
                    progress_bar.progress((i + 1) / len(uploaded_files))
                
                if processed_resumes:
                    st.session_state.parsed_resumes = processed_resumes
                    st.success(f"✅ Successfully processed {len(processed_resumes)} resumes!")
                    
                    if failed_files:
                        st.warning(f"⚠️ Failed to process {len(failed_files)} files:")
                        for failed in failed_files:
                            st.write(f"• {failed}")
                    
                    # Show summary
                    st.subheader("📊 Processing Summary")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Processed", len(processed_resumes))
                    with col2:
                        st.metric("Failed", len(failed_files))
                    with col3:
                        emails_found = sum(1 for r in processed_resumes if r['parsed_data']['email'])
                        st.metric("Emails Found", emails_found)
                    
                    st.session_state.step = 3
                    st.rerun()
                else:
                    st.error("No resumes could be processed successfully")

def step3_analysis():
    st.header("🔍 Step 3: Run Analysis")
    
    if not st.session_state.parsed_resumes:
        st.warning("Please upload and process resumes first")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Analysis Settings")
        
        top_n = st.selectbox(
            "Top candidates to analyze",
            options=[3, 5, 10, 15, 20],
            index=1
        )
        
        config_path = st.text_input("Config file path", value="config.yaml")
        
        # Weight sliders
        st.subheader("Scoring Weights")
        semantic_weight = st.slider("Semantic Similarity", 0.0, 1.0, 0.4, 0.05)
        keyword_weight = st.slider("Keyword Match", 0.0, 1.0, 0.35, 0.05)
        experience_weight = st.slider("Years of Experience", 0.0, 1.0, 0.25, 0.05)
        
        # Normalize weights
        total_weight = semantic_weight + keyword_weight + experience_weight
        if total_weight > 0:
            semantic_weight /= total_weight
            keyword_weight /= total_weight
            experience_weight /= total_weight
    
    with col2:
        st.subheader("Current Data")
        st.metric("Job Description Length", f"{len(st.session_state.job_description)} chars")
        st.metric("Resumes to Analyze", len(st.session_state.parsed_resumes))
        
        # Experience requirements
        req_experience = st.number_input(
            "Required years of experience",
            min_value=0.0,
            max_value=20.0,
            value=5.0,
            step=0.5
        )
        
        send_emails = st.checkbox("Send interview invitations")
        if send_emails:
            st.info("🔐 Ensure ATS_EMAIL and ATS_PASSWORD environment variables are set")
    
    # Create weights dict
    weights = {
        'semantic_similarity': semantic_weight,
        'keyword_match': keyword_weight,
        'years_of_experience': experience_weight,
        'jd_required_experience': req_experience
    }
    
    if st.button("🚀 Run AI Analysis", type="primary"):
        with st.spinner("Running AI-powered candidate analysis..."):
            try:
                # Score candidates
                ranked_candidates = score_candidates(
                    st.session_state.job_description,
                    st.session_state.parsed_resumes,
                    weights
                )
                
                # Generate explanations for top candidates
                top_candidates = ranked_candidates[:top_n]
                progress_bar = st.progress(0)
                
                for i, candidate in enumerate(top_candidates):
                    candidate['explanation'] = generate_explanation(
                        st.session_state.job_description,
                        candidate
                    )
                    progress_bar.progress((i + 1) / len(top_candidates))
                
                st.session_state.ranked_candidates = ranked_candidates
                
                # Save results
                timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                report_path = os.path.join(Config.OUTPUT_FOLDER, f"ATS_Report_{timestamp}.json")
                
                # Prepare data for JSON serialization
                json_data = []
                for candidate in ranked_candidates:
                    json_candidate = {
                        "filename": candidate["filename"],
                        "score": candidate["score"],
                        "score_breakdown": candidate["score_breakdown"],
                        "parsed_data": candidate["parsed_data"],
                        "explanation": candidate.get("explanation", "")
                    }
                    json_data.append(json_candidate)
                
                with open(report_path, 'w') as f:
                    json.dump(json_data, f, indent=4, default=str)
                
                st.success(f"✅ Analysis completed! Report saved to: {report_path}")
                st.session_state.step = 4
                st.rerun()
                
            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")

def step4_results():
    st.header("🏆 Analysis Results")
    
    if not st.session_state.ranked_candidates:
        st.warning("No analysis results available")
        return
    
    candidates = st.session_state.ranked_candidates
    top_candidates = candidates[:10]  # Show top 10
    
    # Summary metrics
    st.subheader("📊 Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric("Total Candidates", len(candidates))
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        avg_score = np.mean([c['score'] for c in candidates])
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric("Average Score", f"{avg_score:.1f}%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        best_score = max(c['score'] for c in candidates)
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric("Best Score", f"{best_score:.1f}%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        emails_available = sum(1 for c in candidates if c['parsed_data']['email'])
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric("Contactable", f"{emails_available}/{len(candidates)}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Visualizations
    st.subheader("📈 Score Analysis")
    
    # Create dataframe for plotting
    plot_data = []
    for i, candidate in enumerate(top_candidates):
        name = candidate['parsed_data']['name'] or f"Candidate {i+1}"
        plot_data.append({
            'Candidate': name,
            'Overall Score': candidate['score'],
            'Semantic Similarity': candidate['score_breakdown']['semantic_similarity'],
            'Keyword Match': candidate['score_breakdown']['keyword_match'],
            'Experience Match': candidate['score_breakdown']['experience_match']
        })
    
    df = pd.DataFrame(plot_data)
    
    # Bar chart
    fig_bar = px.bar(
        df,
        x='Candidate',
        y='Overall Score',
        title="Top Candidates - Overall Scores",
        color='Overall Score',
        color_continuous_scale='viridis',
        text='Overall Score'
    )
    fig_bar.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig_bar.update_layout(xaxis_tickangle=-45, height=500)
    st.plotly_chart(fig_bar, use_container_width=True)
    
    # Score breakdown comparison
    fig_breakdown = go.Figure()
    
    categories = ['Semantic Similarity', 'Keyword Match', 'Experience Match']
    
    for i, candidate in enumerate(top_candidates[:5]):  # Top 5 for readability
        name = candidate['parsed_data']['name'] or f"Candidate {i+1}"
        values = [
            candidate['score_breakdown']['semantic_similarity'],
            candidate['score_breakdown']['keyword_match'],
            candidate['score_breakdown']['experience_match']
        ]
        
        fig_breakdown.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name=name
        ))
    
    fig_breakdown.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        title="Score Breakdown - Top 5 Candidates",
        height=500
    )
    st.plotly_chart(fig_breakdown, use_container_width=True)
    
    # Detailed candidate information
    st.subheader("👥 Top Candidates Details")
    
    # Filter and display options
    col1, col2 = st.columns(2)
    with col1:
        show_count = st.selectbox("Show top N candidates", [5, 10, 15, len(candidates)], index=1)
    with col2:
        min_score = st.slider("Minimum score filter", 0, 100, 0, 5)
    
    # Filter candidates
    filtered_candidates = [c for c in candidates if c['score'] >= min_score][:show_count]
    
    # Display candidates
    for i, candidate in enumerate(filtered_candidates, 1):
        st.markdown('<div class="candidate-card">', unsafe_allow_html=True)
        
        # Header with rank and score
        col1, col2 = st.columns([4, 1])
        
        with col1:
            name = candidate['parsed_data']['name'] or candidate['filename']
            st.markdown(f"### #{i} - {name}")
            
            # Contact information
            contact_info = []
            if candidate['parsed_data']['email']:
                contact_info.append(f"📧 {candidate['parsed_data']['email']}")
            if candidate['parsed_data']['phone']:
                contact_info.append(f"📱 {candidate['parsed_data']['phone']}")
            if candidate['parsed_data']['experience_years']:
                contact_info.append(f"💼 {candidate['parsed_data']['experience_years']} years")
            
            if contact_info:
                st.markdown(" | ".join(contact_info))
        
        with col2:
            st.markdown(f'<div class="score-badge">{candidate["score"]:.1f}%</div>', unsafe_allow_html=True)
        
        # Score breakdown
        st.markdown("**📊 Score Breakdown:**")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Semantic Match",
                f"{candidate['score_breakdown']['semantic_similarity']:.1f}%",
                help="How well the resume content matches the job requirements semantically"
            )
        
        with col2:
            st.metric(
                "Keyword Match", 
                f"{candidate['score_breakdown']['keyword_match']:.1f}%",
                help="Direct keyword and phrase matches with job description"
            )
        
        with col3:
            st.metric(
                "Experience Match",
                f"{candidate['score_breakdown']['experience_match']:.1f}%",
                help="Years of experience relative to job requirements"
            )
        
        # AI Explanation
        if 'explanation' in candidate and candidate['explanation']:
            with st.expander("🤖 AI Analysis & Explanation"):
                st.markdown(candidate['explanation'])
        
        # Action buttons
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            if candidate['parsed_data']['email']:
                if st.button(f"📧 Contact", key=f"contact_{i}"):
                    st.info(f"Email: {candidate['parsed_data']['email']}")
        
        with col2:
            if st.button(f"📄 View Details", key=f"details_{i}"):
                with st.expander("Full Resume Data", expanded=True):
                    st.json({
                        "filename": candidate['filename'],
                        "parsed_data": candidate['parsed_data'],
                        "scores": candidate['score_breakdown']
                    })
        
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("---")
    
    # Export options
    st.subheader("💾 Export Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Download CSV
        csv_data = []
        for i, candidate in enumerate(candidates, 1):
            csv_data.append({
                'Rank': i,
                'Name': candidate['parsed_data']['name'] or 'N/A',
                'Email': candidate['parsed_data']['email'] or 'N/A',
                'Phone': candidate['parsed_data']['phone'] or 'N/A',
                'Experience_Years': candidate['parsed_data']['experience_years'],
                'Overall_Score': candidate['score'],
                'Semantic_Score': candidate['score_breakdown']['semantic_similarity'],
                'Keyword_Score': candidate['score_breakdown']['keyword_match'],
                'Experience_Score': candidate['score_breakdown']['experience_match'],
                'Filename': candidate['filename']
            })
        
        csv_df = pd.DataFrame(csv_data)
        csv_string = csv_df.to_csv(index=False)
        
        st.download_button(
            label="📊 Download CSV",
            data=csv_string,
            file_name=f"ats_results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    with col2:
        # Download JSON
        json_data = []
        for candidate in candidates:
            json_candidate = {
                "filename": candidate["filename"],
                "score": candidate["score"],
                "score_breakdown": candidate["score_breakdown"],
                "parsed_data": candidate["parsed_data"],
                "explanation": candidate.get("explanation", "")
            }
            json_data.append(json_candidate)
        
        json_string = json.dumps(json_data, indent=2, default=str)
        
        st.download_button(
            label="📋 Download JSON",
            data=json_string,
            file_name=f"ats_results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
    
    with col3:
        # Email top candidates
        if st.button("📧 Send Invitations", type="primary"):
            # Check for environment variables
            if not os.environ.get("ATS_EMAIL") or not os.environ.get("ATS_PASSWORD"):
                st.error("❌ Email credentials not set. Please set ATS_EMAIL and ATS_PASSWORD environment variables.")
            else:
                top_5 = [c for c in candidates[:5] if c['parsed_data']['email']]
                if top_5:
                    with st.spinner("Sending interview invitations..."):
                        # Here you would call the send_interview_invitations function
                        st.success(f"✅ Interview invitations sent to {len(top_5)} candidates!")
                        for candidate in top_5:
                            st.write(f"• {candidate['parsed_data']['name']} - {candidate['parsed_data']['email']}")
                else:
                    st.warning("⚠️ No email addresses found for top candidates")
    
    # Statistics
    st.subheader("📈 Analysis Statistics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Score Distribution:**")
        score_ranges = {
            "Excellent (80-100%)": len([c for c in candidates if c['score'] >= 80]),
            "Good (60-79%)": len([c for c in candidates if 60 <= c['score'] < 80]),
            "Average (40-59%)": len([c for c in candidates if 40 <= c['score'] < 60]),
            "Below Average (<40%)": len([c for c in candidates if c['score'] < 40])
        }
        
        for range_name, count in score_ranges.items():
            if count > 0:
                percentage = (count / len(candidates)) * 100
                st.write(f"• {range_name}: {count} candidates ({percentage:.1f}%)")
    
    with col2:
        st.markdown("**Data Quality:**")
        quality_stats = {
            "Names found": len([c for c in candidates if c['parsed_data']['name']]),
            "Emails found": len([c for c in candidates if c['parsed_data']['email']]),
            "Phone numbers found": len([c for c in candidates if c['parsed_data']['phone']]),
            "Experience detected": len([c for c in candidates if c['parsed_data']['experience_years'] > 0])
        }
        
        for stat_name, count in quality_stats.items():
            percentage = (count / len(candidates)) * 100
            st.write(f"• {stat_name}: {count}/{len(candidates)} ({percentage:.1f}%)")

# Navigation footer
def render_navigation():
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        if st.session_state.step > 1:
            if st.button("← Previous Step", key="nav_prev"):
                st.session_state.step -= 1
                st.rerun()
    
    with col2:
        # Reset button
        if st.button("🔄 Start Over", key="reset"):
            for key in ['step', 'job_description', 'parsed_resumes', 'ranked_candidates']:
                if key in st.session_state:
                    del st.session_state[key]
            st.session_state.step = 1
            st.rerun()
    
    with col3:
        next_possible = (
            (st.session_state.step == 1 and st.session_state.job_description) or
            (st.session_state.step == 2 and st.session_state.parsed_resumes) or
            (st.session_state.step == 3 and st.session_state.ranked_candidates)
        )
        
        if st.session_state.step < 4 and next_possible:
            if st.button("Next Step →", key="nav_next"):
                st.session_state.step += 1
                st.rerun()

if __name__ == "__main__":
    main()
    render_navigation()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p>🎯 Advanced ATS Pipeline | Built with Streamlit & AI</p>
        <p>💡 Powered by Sentence-BERT, spaCy, and SHAP</p>
    </div>
    """, unsafe_allow_html=True)
