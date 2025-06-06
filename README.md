# Phishing Classifier: Detect Fraudulent Websites Using Machine Learning

## 🚀 Overview

The **Phishing Classifier** is an end-to-end Machine Learning project designed to detect **phishing websites** and protect users from **financial and personal data loss**. By analyzing various website features, the model can intelligently predict whether a site is **legitimate (1)** or **malicious (-1)**.

Phishing attacks are a major cybersecurity threat, often leading to identity theft and fraud. This project aims to **automate the detection process using ML algorithms**, providing fast and reliable predictions that can be integrated into real-world web filters, browsers, or security tools.

---

## 🎯 Objectives

* Prevent financial scams caused by phishing websites
* Use machine learning to automate phishing detection
* Build a complete ML workflow with model training, evaluation, and deployment

---

## 🧾 Dataset

* **Source:** Public phishing dataset with 30 features
* **Target:** Binary classification → `-1` (Phishing), `1` (Safe)

### 🔍 Feature Categories:

* **URL-Based Features:** Having IP Address, URL Length, Shortening Service
* **Domain-Based Features:** SSL Final State, Domain Registration Length
* **Content-Based Features:** Request URL, Links in Tags, IFrames

---

## 🛠️ Tech Stack

* **Language:** Python
* **IDE:** VS Code
* **Web Framework:** Flask
* **Frontend:** HTML & CSS
* **Database:** MongoDB Atlas
* **Deployment:** Render
* **Cloud Storage:** AWS S3
* **Containerization:** Docker
* **ML Libraries:**

  * Scikit-learn
  * XGBoost
  * Pandas, NumPy, Matplotlib, Seaborn
* **CI/CD:** GitHub Actions *(planned)*

---

## 🧱 Project Structure

```
phishing-classifier/
│
├── src/
│   ├── components/         # Training pipeline modules
│   ├── pipeline/           # Prediction and training pipeline orchestrators
│   ├── data_access/        # MongoDB data fetch helpers
│   ├── configuration/      # MongoDB client setup
│   ├── utils/              # Utility functions and model evaluation
│   ├── constants/          # Constant paths and schema definitions
│   └── exception.py        # Custom exception handler
│
├── templates/              # HTML files for UI
├── static/                 # CSS and static assets
├── Dockerfile              # Docker configuration
├── requirements.txt        # Python dependencies
├── app.py                  # Flask entry point
└── render.yaml             # Render deployment config
```

---

## 🧪 Machine Learning Workflow

### ✅ Training Pipeline:

1. **Data Ingestion**: Fetches data from MongoDB Atlas → stores in `artifacts/raw.csv`
2. **Data Validation**: Checks schema, nulls, column match, etc.
3. **Data Transformation**: Train-test split, encoding
4. **Model Training**: Trains using:

   * XGBoost ✅
   * Logistic Regression
   * Naive Bayes
5. **Model Evaluation**: Saves `model.pkl` with accuracy logging
6. **Model Export**: Uploads final model to AWS S3

### 🔄 Prediction Pipeline:

* Accepts test input via Flask POST request
* Downloads `model.pkl` from S3
* Predicts and returns phishing/safe classification

### 🔥 Accuracy:

* Achieved up to **99% accuracy** with XGBoost

---

## 🌐 Deployment

* ✅ Dockerized using custom `Dockerfile`
* ✅ Deployed to Render as a Web Service
* ✅ Auto-starts Flask app on `0.0.0.0:$PORT`
* 🌐 Public access via Render-generated URL

---

## 📊 Visuals (Optional: Add Images to GitHub)

* EDA Visualizations
* HTML Frontend Screenshot
* Prediction Results

---

## 🧠 HLD (High Level Design)

```plaintext
User → HTML UI → Flask API → Predict Pipeline
                       ↑
             Model (S3) ← Model Trainer ← Data Pipeline ← MongoDB
```

## 🔍 LLD (Low Level Design)

* `MongoDBClient`: Connects and reads datasets
* `DataIngestion`: Pulls data and saves locally
* `DataValidation`: Schema matching, null checks
* `DataTransformation`: Preprocessing and splits
* `ModelTrainer`: Trains multiple models, selects best
* `PredictPipeline`: Loads model and predicts on input
* `AWSUploader`: Uploads model to S3

---

---

## 💡 How to Run Locally

```bash
# Clone the repo
https://github.com/yourusername/phishing-classifier.git
cd phishing-classifier

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate for Windows

# Install dependencies
pip install -r requirements.txt

# Run Flask App
python app.py
```

> App will run on `http://localhost:5000`

---

## ✅ Conclusion

The **Phishing Classifier** successfully demonstrates how Machine Learning can be used to combat real-world cyber threats like phishing. With a robust backend pipeline, clean frontend UI, Dockerization, and cloud deployment, this project showcases both technical depth and practical application.

It not only solves the problem of detecting phishing sites with high accuracy, but is also architected in a modular and scalable fashion, making it ready for production-grade use.

---

## 🚀 Future Improvement

* Enhance frontend with React or Tailwind CSS
* Add alert/notification system on detection

---

## 🤝 Connect

Made with 💻 by Dev Pandey.

> Feel free to fork ⭐ and contribute to the project
