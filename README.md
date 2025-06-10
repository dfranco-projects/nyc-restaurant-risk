# NYC Restaurant Inspection Risk Classification

This project regards a case study to develop a classification model to predict health and hygiene **Criticality** (Critical/Not Critical) for restaurants in New York City based on inspection and violation open-sourced data. It also integrates **generative AI** to produce mock public health alerts.

## 🔍 Problem Overview

The NYC Health Department inspects restaurants regularly and assigns them a critical flag based on observed violations. Early identification of high-risk establishments can help prioritize enforcement and protect public health.

## 📂 Project Structure

```bash
nyc-restaurant-risk/
│
├── data/                   # raw and processed data
│   ├── raw/                # raw dataset
│   └── processed/          # cleaned + split datasets
│
├── notebooks/              # jupyter workflows
│   ├── 01_data_understanding.ipynb
│   ├── 02_preprocessing_eda.ipynb
│   ├── 03_feature_eng_modeling.ipynb
│   ├── 04_evaluation_visualization.ipynb
│   └── 05_genai_alerts.ipynb
│
├── src/                     # utility/reusable functions
│   ├── data_utils.py
│   ├── eda_utils.py
│   ├── modeling.py
│   └── genai.py
│
├── model/                  # final model and metrics
├── plots/                  # Confusion matrix, maps, EDA figures
├── slides/                 # Final presentation
├── README.md               # project overview
└── requirements.txt        # python dependencies
```


## ⚙️ Tech Stack

- Python 3.10+
- Jupyter Notebooks & Python Scripts
- Pandas, NumPy, nltk
- Scikit-learn, Optuna, Hugging Face
- Seaborn, Matplotlib

## 🚀 How to Run Locally

1. **Clone the repo**

```bash
git clone https://github.com/dfranco-projects/nyc-restaurant-risk.git
cd nyc-restaurant-risk
```

2. **Create virtual environment**

```bash
python3 -m venv .venv
source .venv/bin/activate        # On Windows: .venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Run notebooks in order**

Open Jupyter or VS Code and start with:

- `01_data_understanding.ipynb`
- `02_eda.ipynb`
- `03_feature_engineering.ipynb`
- `04_modeling.ipynb`
- `05_insights_gen_ai.ipynb`

Each notebook builds on the previous step and produces artifacts (e.g., processed data, model outputs).
