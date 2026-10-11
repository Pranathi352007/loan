# Loan Explainable AI

A simple FastAPI + HTML/CSS/JavaScript project for two workflows, using the checked-in **synthetic** loan dataset, a Gradient Boosting model, SHAP, and Gemini for applicant-friendly wording. This is an educational prototype—not a bank decision system.

## The two options

### 1. Predict a new loan

Enter the financial and application features on the frontend. The backend:
1. Uses the saved Gradient Boosting pipeline to estimate Approved/Rejected.
2. Uses feature-contribution analysis internally to select up to three details related to the estimate.
3. Sends those details and pre-checked suggestions to Gemini for simple, plain-English wording.
4. Shows the applicant's values, short reasons, and suggestions when the estimate is Rejected. It does not show raw SHAP values or technical contribution bars.

### 2. Explain an existing application by ID

Enter a `loan_id` from the checked-in CSV. The backend:
1. Looks up the record in `synthetic_loan_dataset (1).csv` (or `data/loan_dataset.csv` as fallback).
2. Displays the row's stored `loan_status` separately from the Gradient Boosting model's estimate.
3. Uses internal feature-contribution analysis to select details related to the model's estimate, then asks Gemini to write simple reasons.
4. Shows suggestions when the dataset row's stored status is Rejected. The screen does not expose numerical SHAP contributions.

**Important:** the dataset stores an outcome label, not a bank's written reasons. The feature-contribution analysis is used internally to select details; Gemini explains those details in simple language. These explanations describe the demo model's estimate, not the actual internal reasoning of a bank. If the stored decision and model estimate differ, the UI flags the mismatch. The checked-in data is synthetic; these IDs are for the demo only.

## Gemini setup (keep the key private)

1. Create a Gemini API key in Google's API/key console.
2. In the project root—the same folder as `train_model.py`—create a file named `.env`.
3. Copy the template below into `.env` and replace the placeholder with your own key:

```dotenv
GEMINI_API_KEY=replace_with_your_real_key
GEMINI_MODEL=gemini-3.5-flash-lite
```

You can copy `.env.example` as a starting point. The key is read by the backend only; it is not sent to the browser. **Never commit `.env`, paste the key into Python/JavaScript, or share the key in GitHub.** The repository's `.gitignore` excludes `.env`. If a key was ever committed, revoke it and generate a replacement.

If `GEMINI_API_KEY` is missing or the Gemini request fails, the app shows short, safe fallback explanations and the standard suggestions instead of raw SHAP values. The status line tells you that Gemini is not configured. You can change the model with `GEMINI_MODEL` without editing source code.

## Install and run on Windows

Use Python 3.11 (the serialized scikit-learn model and scientific dependencies may not work with every Python version).

Open PowerShell in the project root:

```powershell
py -3.11 -m venv .venv311
.\.venv311\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Regenerate the model and reports in the same environment if the checked-in model artifact cannot load, or after the training script changes:

```powershell
python train_model.py
```

Start the backend and serve the frontend:

```powershell
python -m uvicorn backend.main:app --reload
```

Open <http://127.0.0.1:8000>. API docs: <http://127.0.0.1:8000/docs>.

Run tests:

```powershell
python -m pytest -q
```

## Project structure

```text
.
├── backend/
│   ├── main.py                 # FastAPI and the two workflows
│   ├── model_service.py        # Loads Gradient Boosting pipeline and predicts
│   ├── explanation_service.py  # Individual SHAP contributions
│   ├── gemini_service.py       # Secure Gemini natural-language explanation
│   ├── suggestion_service.py   # Conservative educational suggestions
│   └── schemas.py              # Request/response validation
├── frontend/
│   ├── index.html              # Two workflow buttons
│   ├── styles.css
│   └── app.js
├── models/                     # Saved pipeline and metadata
├── data/
├── reports/
├── tests/
├── .env.example                # Template only; no secret
├── train_model.py
└── requirements.txt
```

## Main endpoints

- `GET /api/health` — model and Gemini configuration status; never returns the API key.
- `GET /api/schema` — form fields based on the trained model schema.
- `POST /api/predict-and-explain` — predict and explain a new application.
- `GET /api/application/{application_id}` — find a stored row by `loan_id` and explain the model's view of it.
- `POST /api/predict` — prediction-only API.
- `POST /api/explain` — applicant-friendly explanation in the same format as the main workflow.
- `GET /api/dependency-analysis` — feature dependency report.
- `GET /api/evaluation` — model metrics.

## Interpretation and safety

- The applicant-facing screen shows the status and simple explanations, not prediction probabilities or numerical SHAP values.
- Feature-contribution analysis is used internally to choose the strongest details. The explanations do not prove causation or reveal a bank's written reason.
- The existing-ID workflow searches the local CSV. It does not connect to a real bank database.
- Gemini writes plain-language reasons from selected details and only rephrases pre-computed suggestions. It does not choose the predicted class or compute feature contributions.
- Synthetic data and labels cannot establish real-world underwriting validity or fairness.
- Do not commit API keys, private applicant data, or real customer records to Git.
