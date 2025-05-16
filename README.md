# ResumeGenius-AI 📄

An intelligent resume screening application that uses Machine Learning to automatically classify resumes into different job categories. Built with Python, Streamlit, and Scikit-learn.

## 🌟 Features

- Upload resumes in multiple formats (PDF, DOCX, TXT)
- Automatic resume validation and verification
- Smart category prediction with confidence scores
- Beautiful and intuitive user interface
- Detailed document analysis
- Support for multiple job categories
- Real-time processing and results

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/pravinboopathi/SmartCV-Classifier.git
cd SmartCV-Classifier
```

2. Create and activate a virtual environment:
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

3. Install required packages:
```bash
pip install streamlit scikit-learn python-docx PyPDF2 pandas numpy nltk
```

### Running the Application

1. First, train the model (only needed once):
```bash
python train_model.py
```

2. Start the Streamlit app:
```bash
streamlit run app.py
```

3. Open your browser and go to:
```
http://localhost:8501
```

## 📊 How It Works

1. **Upload Resume**: Submit your resume in PDF, DOCX, or TXT format
2. **Document Analysis**: The app validates if it's a proper resume by checking:
   - Resume-specific keywords
   - Section headers
   - Document length and structure
3. **Category Prediction**: For valid resumes, the app:
   - Extracts and processes text
   - Analyzes content using ML model
   - Predicts the most suitable job category
   - Provides a confidence score

## 🎯 Supported Categories

The model can classify resumes into various job categories including:
- Software Development
- Web Development
- HR
- Sales
- Marketing
- Data Science
- And more...

## 📝 Usage Tips

1. Ensure your resume is properly formatted
2. Include relevant sections (Education, Experience, Skills)
3. Use clear and standard section headers
4. Make sure the document is text-extractable
5. Keep the resume length reasonable (2-5 pages)

## ⚠️ Troubleshooting

If you encounter any issues:

1. **Model not found error**:
   - Ensure you've run `train_model.py` before starting the app
   - Check if `clf.pkl`, `tfidf.pkl`, and `encoder.pkl` exist in the root directory

2. **Package not found errors**:
   - Make sure you're in the virtual environment
   - Try reinstalling requirements: `pip install -r requirements.txt`

3. **PDF extraction issues**:
   - Ensure the PDF is not scanned/image-based
   - Check if the PDF is text-searchable

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built with Streamlit
- Powered by Scikit-learn
- Uses NLTK for text processing
- Inspired by the need for efficient resume screening 
