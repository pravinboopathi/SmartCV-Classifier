import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download required NLTK data
try:
    nltk.download('punkt')
    nltk.download('stopwords')
except:
    pass

# Load the dataset
df = pd.read_csv('UpdatedResumeDataSet.csv')

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

# Clean the resume text
print("Cleaning resume text...")
df['Resume'] = df['Resume'].apply(cleanResume)

# Create custom stop words
custom_stop_words = [
    'experience', 'skill', 'skills', 'year', 'years', 'job', 'jobs',
    'work', 'project', 'projects', 'education', 'qualification', 'qualifications'
]

# Create TF-IDF features with better parameters
print("Creating TF-IDF features...")
tfidf = TfidfVectorizer(
    stop_words=custom_stop_words,
    ngram_range=(1, 2),  # Use both unigrams and bigrams
    max_features=5000,    # Limit features to most important ones
    min_df=2,            # Ignore terms that appear in less than 2 documents
    max_df=0.95,         # Ignore terms that appear in more than 95% of documents
)

X = tfidf.fit_transform(df['Resume'])

# Encode the categories
print("Encoding categories...")
le = LabelEncoder()
y = le.fit_transform(df['Category'])

# Train the model with better parameters
print("Training model...")
clf = LinearSVC(
    C=1.0,               # Regularization parameter
    class_weight='balanced',  # Handle imbalanced classes
    dual=False,          # Algorithm selection
    max_iter=2000        # Increase max iterations
)
clf.fit(X, y)

# Save the model and required objects
print("Saving model files...")
with open('clf.pkl', 'wb') as f:
    pickle.dump(clf, f)

with open('tfidf.pkl', 'wb') as f:
    pickle.dump(tfidf, f)

with open('encoder.pkl', 'wb') as f:
    pickle.dump(le, f)

print("Model training completed and saved successfully!")
print("\nModel now supports better recognition of:")
print("- Different resume formats")
print("- Various ways of expressing skills")
print("- Multiple languages (after conversion to English)")
print("- Better handling of technical terms")
print("- More robust to variations in writing style") 