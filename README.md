# NYC Restaurant Inspection Risk Classification

This project builds a multiclass classification model to predict health **risk categories** (Low, Moderate, High) for restaurants in New York City based on inspection and violation data. It also integrates **generative AI** to produce mock public health alerts.

## 🔍 Problem Overview

The NYC Health Department inspects restaurants regularly and assigns them a risk category based on observed violations. Early identification of high-risk establishments can help prioritize enforcement and protect public health.

## 📂 Project Structure



## ⚙️ Tech Stack

- Python 3.10+
- Jupyter Notebooks
- Scikit-learn
- Pandas, NumPy
- XGBoost, LightGBM
- Seaborn, Matplotlib, Plotly
- GeoPandas (for maps)
- OpenAI API (for bonus GenAI task)

## 🚀 How to Run Locally

1. **Clone the repo**

```bash
git clone https://github.com/dfranco-projects/nyc-restaurant-risk.git
cd nyc-restaurant-risk
```

2. **Create virtual environment**

```bash
python -m venv .venv
source .venv/bin/activate        # On Windows: .venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Run notebooks in order**

Open Jupyter or VS Code and start with:

- `01_data_understanding.ipynb`
- `02_preprocessing_eda.ipynb`
- `03_feature_eng_modeling.ipynb`
- `04_evaluation_visualization.ipynb`
- `05_genai_alerts.ipynb` (optional bonus)

Each notebook builds on the previous step and produces artifacts (e.g., processed data, model outputs).

## 🧠 Bonus: Generative AI Alerts

In , we use OpenAI's GPT to generate automated public health alerts based on borough-level trends in high-risk violations.

---

## 📈 Deliverables

- Clean, well-commented notebooks
- Model performance metrics (accuracy, macro-F1)
- Visualizations (EDA, confusion matrix, borough heatmaps)
- Presentation slide deck (5–7 slides)
- README with full reproducibility instructions

