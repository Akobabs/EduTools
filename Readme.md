### **TALE Predictive Framework**

A machine learning-based educational analytics tool designed to predict student performance in **Technology-Assisted Learning Environments (TALEs)**. It leverages the **OULAD dataset** and includes **explainable AI** through SHAP to ensure transparent and interpretable predictions.

---

### 🚀 Features

* Predict academic performance using ML models trained on the OULAD dataset.
* Explainable AI insights using SHAP visualizations.
* Web interface powered by Flask.
* Modular project structure with Jupyter Notebooks for EDA and model training.

---

### ⚙️ Setup Instructions

1. **Clone the repository**

   ```bash
   git clone <your-repo-url>
   cd <your-repo-folder>
   ```

2. **Create and activate a virtual environment**

   * **Linux/macOS**:

     ```bash
     python -m venv venv
     source venv/bin/activate
     ```
   * **Windows**:

     ```bash
     python -m venv venv
     venv\Scripts\activate
     ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Download the OULAD dataset**

   * Source: [OULAD Dataset](https://analyse.kmi.open.ac.uk/open_dataset)
   * Place CSV files in: `data/raw/`

---

### 🧠 Usage

#### Option 1: Jupyter Notebook (for analysis and training)

1. Launch Jupyter:

   ```bash
   jupyter notebook
   ```

2. Open and run:

   ```
   notebooks/student_performance_prediction.ipynb
   ```

   * Preprocess the dataset
   * Train machine learning models
   * Generate and interpret SHAP visualizations

---

#### Option 2: Flask Web App (for deployment)

1. Ensure trained models are saved in `models/`
2. Run the app:

   ```bash
   python src/app.py
   ```
3. Access the app at:

   ```
   http://localhost:5000
   ```

---

### 📁 Project Structure

```
TALE_Predictive_Framework/
│
├── data/                # Raw and processed OULAD data
│   └── raw/
├── src/                 # Core scripts (preprocessing, modeling, Flask app)
├── templates/           # HTML templates for Flask UI
├── notebooks/           # Jupyter notebooks for EDA and modeling
├── tests/               # Unit tests
├── models/              # Saved/trained model files
├── figures/             # Generated plots and SHAP visuals
└── docker/              # Docker setup for deployment
```

---

### 🪪 License

This project is licensed under the **MIT License** — feel free to use and modify it.

---