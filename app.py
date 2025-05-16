# you need to install all these in your terminal
# pip install streamlit
# pip install scikit-learn
# pip install python-docx
# pip install PyPDF2


import streamlit as st
import pickle
import docx  # Extract text from Word file
import PyPDF2  # Extract text from PDF
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np

# Download required NLTK data
try:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')
except:
    pass

# Load pre-trained model and vectorizer
try:
    svc_model = pickle.load(open('clf.pkl', 'rb'))
    tfidf = pickle.load(open('tfidf.pkl', 'rb'))
    le = pickle.load(open('encoder.pkl', 'rb'))
except Exception as e:
    st.error("Error loading model files. Please make sure you have run train_model.py first.")
    st.stop()

# Define resume-specific keywords for validation
RESUME_KEYWORDS = {
    'education': ['education', 'university', 'college', 'school', 'degree', 'bachelor', 'master', 'phd', 'diploma'],
    'experience': ['experience', 'work', 'job', 'career', 'employment', 'position', 'role'],
    'skills': ['skills', 'abilities', 'expertise', 'proficient', 'knowledge', 'competencies'],
    'contact': ['email', 'phone', 'address', 'contact', 'linkedin', 'github'],
    'sections': ['summary', 'objective', 'profile', 'projects', 'achievements', 'certifications']
}

def is_valid_resume(text):
    """Check if the document appears to be a resume."""
    text = text.lower()
    
    # Count resume keywords found in the text
    keyword_count = 0
    section_count = 0
    
    for category, keywords in RESUME_KEYWORDS.items():
        for keyword in keywords:
            if keyword in text:
                keyword_count += 1
                if category == 'sections':
                    section_count += 1
    
    # Calculate text statistics
    words = text.split()
    word_count = len(words)
    
    # Validation criteria:
    # 1. Should have minimum number of resume keywords
    # 2. Should have at least some section keywords
    # 3. Should have reasonable length
    # 4. Should not be too short or too long
    is_valid = (
        keyword_count >= 5 and  # At least 5 resume-related keywords
        section_count >= 2 and  # At least 2 section headers
        100 <= word_count <= 5000  # Reasonable resume length
    )
    
    return is_valid, {
        'keyword_count': keyword_count,
        'section_count': section_count,
        'word_count': word_count
    }

# Function to clean resume text
def cleanResume(txt):
    if not isinstance(txt, str):
        return ""
    
    # Convert to lowercase
    txt = txt.lower()
    
    # Remove URLs
    txt = re.sub(r'http\S+\s*', ' ', txt)
    
    # Remove RT and cc
    txt = re.sub(r'rt|cc', ' ', txt)
    
    # Remove hashtags and mentions
    txt = re.sub(r'[#@]\S+', ' ', txt)
    
    # Remove punctuations and special characters
    txt = re.sub(r'[^\w\s]', ' ', txt)
    
    # Remove non-ASCII chars
    txt = re.sub(r'[^\x00-\x7f]', ' ', txt)
    
    # Remove extra whitespace
    txt = re.sub(r'\s+', ' ', txt)
    
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    words = txt.split()
    words = [w for w in words if w not in stop_words]
    
    return ' '.join(words).strip()


# Function to extract text from PDF
def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ''
    for page in pdf_reader.pages:
        text += page.extract_text() + ' '
    return text


# Function to extract text from DOCX
def extract_text_from_docx(file):
    doc = docx.Document(file)
    text = ''
    for paragraph in doc.paragraphs:
        text += paragraph.text + ' '
    return text


# Function to extract text from TXT with explicit encoding handling
def extract_text_from_txt(file):
    try:
        text = file.read().decode('utf-8')
    except UnicodeDecodeError:
        try:
            text = file.read().decode('latin-1')
        except:
            text = file.read().decode('utf-8', errors='ignore')
    return text


# Function to handle file upload and extraction
def handle_file_upload(uploaded_file):
    if uploaded_file is None:
        return None
        
    file_extension = uploaded_file.name.split('.')[-1].lower()
    
    try:
        if file_extension == 'pdf':
            text = extract_text_from_pdf(uploaded_file)
        elif file_extension == 'docx':
            text = extract_text_from_docx(uploaded_file)
        elif file_extension == 'txt':
            text = extract_text_from_txt(uploaded_file)
        else:
            st.error("Unsupported file type. Please upload a PDF, DOCX, or TXT file.")
            return None
            
        return text
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return None


# Function to predict the category of a resume
def predict_category(text):
    # First check if it's a valid resume
    is_valid, stats = is_valid_resume(text)
    
    if not is_valid:
        return None, 0, stats
    
    # Clean and preprocess the text
    cleaned_text = cleanResume(text)
    
    # Vectorize the text
    features = tfidf.transform([cleaned_text])
    
    # Make prediction
    prediction = svc_model.predict(features)
    
    # Get category name
    category = le.inverse_transform(prediction)[0]
    
    # Get prediction confidence
    decision = svc_model.decision_function(features)
    confidence = float(abs(decision[0]).max())  # Convert to float and get max value
    
    return category, confidence, stats


# Streamlit app layout
def main():
    st.set_page_config(page_title="Resume Category Prediction", page_icon="📄", layout="wide")

    # Custom CSS for better UI
    st.markdown("""
        <style>
        .main {
            padding: 2rem;
        }
        .stButton>button {
            width: 100%;
            margin-top: 1rem;
        }
        .upload-text {
            text-align: center;
            padding: 2rem;
            border: 2px dashed #cccccc;
            border-radius: 5px;
        }
        .stAlert {
            padding: 1rem;
            margin-bottom: 1rem;
            border-radius: 0.5rem;
        }
        .stats-box {
            background-color: #f8f9fa;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 1rem 0;
        }
        .invalid-doc {
            color: #dc3545;
            padding: 1rem;
            background-color: #f8d7da;
            border-radius: 0.5rem;
            margin: 1rem 0;
        }
        </style>
    """, unsafe_allow_html=True)

    # Header
    st.title("📄 Resume Category Prediction")
    st.markdown("""
    Upload your resume (PDF, DOCX, or TXT format) and get the predicted job category.
    The model will analyze the content and suggest the most suitable job category.
    """)

    # File upload section
    uploaded_file = st.file_uploader("Choose a Resume", type=["pdf", "docx", "txt"])

    if uploaded_file is not None:
        with st.spinner("Processing document..."):
            # Extract text from the uploaded file
            resume_text = handle_file_upload(uploaded_file)
            
            if resume_text:
                st.success("Successfully extracted text from the document.")
                
                # Show extracted text if user wants to see it
                if st.checkbox("Show extracted text", False):
                    st.text_area("Extracted Text", resume_text, height=200)
                
                # Make prediction
                category, confidence, stats = predict_category(resume_text)
                
                # Display document statistics
                st.markdown("### 📊 Document Analysis")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Resume Keywords", stats['keyword_count'])
                with col2:
                    st.metric("Section Headers", stats['section_count'])
                with col3:
                    st.metric("Word Count", stats['word_count'])
                
                if category is None:
                    st.markdown("""
                    <div class='invalid-doc'>
                        <h3>⚠️ This document does not appear to be a resume</h3>
                        <p>The uploaded document lacks typical resume characteristics:</p>
                        <ul>
                            <li>Missing essential resume sections (education, experience, skills, etc.)</li>
                            <li>Insufficient resume-related keywords</li>
                            <li>Unusual document length for a resume</li>
                        </ul>
                        <p>Please ensure you're uploading a proper resume document.</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    # Display results
                    st.markdown("### 🎯 Prediction Results")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown(f"**Predicted Category:** {category}")
                    
                    with col2:
                        confidence_percentage = min(100, confidence * 100)
                        st.markdown(f"**Confidence Score:** {confidence_percentage:.2f}%")
                    
                    # Additional information
                    st.markdown("### 📊 Category Information")
                    st.markdown(f"""
                    The resume appears to be most suitable for the **{category}** category. 
                    This prediction is based on:
                    - Key terms and skills found in the resume
                    - Overall content and context
                    - Comparison with similar profiles
                    """)
                    
                    # Warning for low confidence predictions
                    if confidence_percentage < 50:
                        st.warning("""
                        ⚠️ Note: The confidence score is relatively low. This might be because:
                        - The resume format is significantly different from our training data
                        - The content might span multiple categories
                        - Some key information might not be properly extracted
                        
                        Consider reviewing and updating your resume with more specific keywords related to your target role.
                        """)


if __name__ == "__main__":
    main()
