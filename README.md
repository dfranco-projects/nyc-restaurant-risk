# NYC Restaurant Inspection Risk Classification

## 🔍 Problem Overview

The NYC Health Department inspects restaurants regularly and assigns them a critical flag based on observed violations. Early identification of high-risk establishments can help prioritize enforcement and protect public health.

## 📂 Project Structure

```bash
nyc-restaurant-risk/
│
├── data/                   # raw and processed data
├── notebooks/              # jupyter workflows
│   ├── 01_data_understanding.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_modeling.ipynb
│   └── 05_genai_alerts.ipynb
│
├── src/                    # utility/reusable functions
├── plots/                  # Maps
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
