# Loan Explainable AI

An educational project that pairs a **Gradient Boosting** loan-status estimate with local **SHAP** explanations, responsible financial suggestions, and a feature-dependency audit. The bundled dataset is synthetic, so model outputs are not real-world lending evidence and must not be used to make actual credit decisions.

## What is included

- **Loan prediction:** approval/rejection output, estimated probabilities, and the selected model threshold.
- **Reasons & suggestions:** individual SHAP contributions, strongest positive/negative signals, and conditional educational suggestions.
- **Feature dependency:** saved global importance, feature-removal/ablation results, and cross-validation ranking stability.
- **FastAPI backend:** API endpoints documented at `/docs`.
- **HTML/CSS/JavaScript frontend:** served by the same FastAPI app; no Streamlit is used.

## Project structure

```text
.
├── backend/
│   ├── main.py                 # FastAPI routes and static UI hosting
│   ├── model_service.py        # Loads the saved pipeline and predicts
│   ├── explanation_service.py  # Individual SHAP explanations
│   ├── suggestion_service.py   # Responsible suggestions
│   └── schemas.py              # Pydantic request/response models
├── data/                       # Dataset copy used in the repository
├── frontend/
│   ├── index.html
│   ├── styles.css
│   └── app.js
├── models/                     # Trained pipeline and metadata
├── reports/                    # Evaluation and dependency-analysis reports
├── tests/test_api.py
├── train_model.py
└── requirements.txt
```

## Run on Windows PowerShell

From the repository root (the folder containing `train_model.py`):

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If model artifacts are missing or incompatible with the installed scikit-learn version, regenerate them from the checked-in synthetic dataset:

```powershell
python train_model.py
```

Then start the application:

```powershell
uvicorn backend.main:app --reload
```

Open <http://127.0.0.1:8000>. API documentation is available at <http://127.0.0.1:8000/docs>.

Run basic tests with:

```powershell
pytest -q
```

## API routes

- `GET /api/health` — artifact/setup status.
- `GET /api/schema` — input feature names and categorical options from model metadata.
- `GET /api/model-info` — model schema, threshold, metrics, and synthetic-data disclaimer.
- `POST /api/predict` — input: the feature object; output: status and class probabilities.
- `POST /api/explain` — input: the feature object; output: SHAP factors and suggestions.
- `GET /api/dependency-analysis` — global feature-dependency and ablation report.
- `GET /api/evaluation` — model evaluation report.

## Important interpretation notes

1. A probability is a model estimate, not a guarantee or a calibrated real-world probability.
2. SHAP explains how the trained model produced an output. It does not show causality, an actual bank's internal reasoning, or proof that a factor is appropriate for underwriting.
3. The feature-removal experiment is a diagnostic of model reliance. Performance loss alone does not establish discrimination or causation.
4. Synthetic labels reflect their generation process. Real deployment would require authorised representative data, independent validation, security controls, privacy protections, regulatory review, and human oversight.
5. Do not commit credentials, private applicant data, or real customer records to Git.
