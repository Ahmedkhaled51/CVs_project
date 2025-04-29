# Resume Categorization Project

## 📌 Overview
This project tackles the challenge of **resume categorization** using machine learning and deep learning techniques. Companies often face the daunting task of sifting through numerous resumes for each job opening. This app aims to automate and streamline this process by predicting the job category a given resume belongs to. By using a trained model, the app can quickly suggest the appropriate job category for each resume, saving time and resources for recruiters.

## 🛠️ Technologies Used
- **Python**
- **Scikit-learn** (Machine Learning Models: KNN, SVC, Random Forest)
- **TensorFlow/Keras** (Deep Learning Models: MLP, RNN, LSTM, BI-LSTM)
- **Streamlit** (Web Application)
- **NLTK/Regex** (Text Preprocessing)
- **Pandas/Numpy** (Data Handling)
- **PyPDF2/python-docx** (File Parsing)

## 📂 Project Structure
```
Resume-Categorization/
├── app/                      # Streamlit application files
│   ├── main.py               # Main Streamlit app script
│   ├── knn.pkl               # Serialized KNN model
│   ├── svc.pkl               # Serialized SVC model
│   ├── rf.pkl                # Serialized Random Forest model
│   ├── mlp.pkl               # Serialized MLP model
│   ├── ensemble.pkl          # Serialized Ensemble model
│   ├── tfidf.pkl             # Serialized TF-IDF vectorizer
│   └── encoder.pkl           # Serialized Label Encoder
├── notebooks/                # Jupyter notebooks for development
│   └── Resume_Categorization.ipynb  # Main project notebook
├── data/                     # Dataset files
│   └── UpdatedResumeDataSet.csv     # Resume dataset
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## 🚀 Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/Resume-Categorization.git
   cd Resume-Categorization
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 🏃‍♂️ Running the Application
To run the Streamlit application:
```bash
streamlit run app/main.py
```

## 📊 Models Implemented
### Machine Learning Models
- **K-Nearest Neighbors (KNN)**
- **Support Vector Machine (SVC)**
- **Random Forest**

### Deep Learning Models
- **Multilayer Perceptron (MLP)**
- **Recurrent Neural Network (RNN)**
- **Long Short-Term Memory (LSTM)**
- **Bidirectional LSTM (BI-LSTM)**

### Ensemble Model
- **Voting Classifier** (Combines KNN, SVC, and Random Forest)

## 📝 Dataset
The project uses the [Resume Dataset](https://www.kaggle.com/datasets/gauravduttakiit/resume-dataset) from Kaggle. This dataset consists of resumes categorized into 25 distinct job fields.

## 🧠 Key Features
1. **Text Preprocessing Pipeline**
   - URL removal
   - Special character removal
   - Whitespace normalization
   - Stopword removal

2. **Multiple Model Comparison**
   - Compare predictions from different models
   - View accuracy metrics for each approach

3. **File Upload Support**
   - Accepts PDF, DOCX, and TXT files
   - Automatic text extraction

4. **Interactive Web Interface**
   - Easy-to-use file upload
   - Clear results display
   - Model performance comparison

## 👥 Team Members
- [Abdelrahman Mahmoud](https://abdelrahmanmah.github.io/SafeZoneInc/Abdelrahman.html)
- [GannaTullah Gouda](https://gannaasaad.github.io/index.html)
- [Ahmed Khaled](https://Ahmedkhaled51.github.io/)
- [Ali Mohamed](https://aliiimohamedaliii.github.io/My-portfolio/)

## 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements.

---

🌟 **Happy resume categorizing!** 🌟
