"""Create simple, factor-specific loan reasons using Gemini.

Numerical feature-contribution scores are used internally by the application and are not
sent to the applicant-facing UI. Prompts receive only the selected feature, displayed name,
the applicant value (for context), and whether the feature supported or hurt the assessment.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)
ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")


class GeminiExplanationService:
    """Generate one plain-language reason per factor, and suggestions only for rejection."""

    _client: Any = None
    _client_key: str | None = None

    @classmethod
    def _get_client(cls) -> Any | None:
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key:
            return None
        if cls._client is None or cls._client_key != api_key:
            cls._client = genai.Client(api_key=api_key)
            cls._client_key = api_key
        return cls._client

    @staticmethod
    def _friendly_name(factor: dict[str, Any]) -> str:
        return str(factor.get("display_name") or str(factor.get("feature", "")).replace("_", " ").title())

    @classmethod
    def _fallback_reason(cls, factor: dict[str, Any], approved: bool) -> str:
        name = cls._friendly_name(factor)
        effect = "positively affected" if approved else "negatively affected"
        return f"{name} {effect} the loan assessment."

    @classmethod
    def _fallback_suggestion(cls, factor: dict[str, Any]) -> str:
        """Use conservative advice tied to the given feature if Gemini is unavailable."""
        feature = str(factor.get("feature", "")).strip().lower()
        if feature == "cibil_score":
            return "Check your credit report for errors and try to make payments on time."
        if feature in {"loan_amount", "loantoIncome".lower()}:
            return "Check whether a smaller loan amount would be easier to repay."
        if feature == "income_annum":
            return "Compare the expected loan payment with your regular income and expenses."
        if feature == "loan_term":
            return "Ask the lender whether another repayment period may better fit your budget."
        if "asset" in feature:
            return "Check that the asset details and supporting documents in your application are correct."
        if feature in {"education", "self_employed", "no_of_dependents"}:
            return "Check that this part of your application is accurate and ask the lender which documents are needed."
        return "Ask the lender what information may be helpful for a future application."

    @classmethod
    def _fallback(
        cls,
        factors: list[dict[str, Any]],
        reason: str,
        approved: bool,
    ) -> dict[str, Any]:
        """Keep the experience useful without showing technical scores if Gemini is offline."""
        reasons = [
            {
                "feature": str(factor.get("feature", "")),
                "explanation": cls._fallback_reason(factor, approved),
            }
            for factor in factors
        ]
        suggestions = [] if approved else [
            {
                "feature": str(factor.get("feature", "")),
                "suggestion": cls._fallback_suggestion(factor),
            }
            for factor in factors
        ]
        return {
            "available": False,
            "overview": "",
            "factor_explanations": reasons,
            "suggestions": suggestions,
        }

    @staticmethod
    def _parse_array(response_text: str) -> list[dict[str, Any]]:
        parsed = json.loads((response_text or "").strip())
        if not isinstance(parsed, list):
            raise ValueError("Gemini returned a JSON value that was not an array.")
        return [item for item in parsed if isinstance(item, dict)]

    @classmethod
    def _build_reason_prompt(
        cls,
        *,
        approved: bool,
        factors_json: str,
    ) -> str:
        decision = "APPROVED" if approved else "REJECTED"
        effect = "positively" if approved else "negatively"
        return f"""
You are an AI loan explanation assistant.

The applicant's loan application was {decision}.

For EACH factor provided below, write ONE simple explanation.

IMPORTANT LANGUAGE RULES:

1. Use VERY SIMPLE ENGLISH.
2. Write for a person who knows only basic English.
3. Use short and clear sentences.
4. Use common everyday words.
5. Avoid financial jargon and difficult words.
6. Explain each factor independently.
7. Do NOT compare one factor with another.
8. Do NOT combine factors.
9. Do NOT introduce any factor that is not provided.
10. Do NOT mention SHAP, machine learning, model, algorithm, prediction, or feature importance.
11. Do NOT say that a factor definitely caused the {decision.lower()} decision.
12. Explain that the factor {effect} affected the loan assessment.
13. Do NOT repeat the applicant's numerical value.
14. The applicant's actual value will be displayed separately.
15. Do NOT give recommendations.
16. Keep each explanation to ONE or TWO short sentences.
17. Do NOT use difficult financial words.
18. Return EXACTLY one explanation for EACH provided factor.

IMPORTANT:
Each explanation MUST be connected to the EXACT feature name provided.
Copy each "feature" value exactly as it appears in the input.

Return ONLY valid JSON. Do not add markdown or any text outside the JSON array.

Use exactly this format:
[
  {{
    "feature": "exact feature name",
    "reason": "simple explanation"
  }}
]

The factors are:
{factors_json}
""".strip()

    @classmethod
    def _build_suggestion_prompt(cls, factors_json: str) -> str:
        return f"""
You are a loan guidance assistant.

The applicant's loan application was REJECTED.

For EACH factor provided below, give ONE practical suggestion.

IMPORTANT RULES:

1. Use VERY SIMPLE ENGLISH.
2. Write for a person who knows only basic English.
3. Use short and clear sentences.
4. Use common everyday words.
5. Each suggestion must directly relate to its EXACT factor.
6. Do NOT give suggestions about any other factor.
7. Do NOT compare the factors.
8. Do NOT combine the factors.
9. Do NOT use technical terms such as SHAP, machine learning, model, algorithm, or feature importance.
10. Do NOT promise that following a suggestion will guarantee approval.
11. Do NOT say that changing a factor will definitely result in approval.
12. For cibil_score, suggest checking the credit report, paying on time, and managing existing debt.
13. For LoanToIncome or loan_amount, suggest considering an amount that is more manageable compared with income.
14. For loan_term, suggest discussing different loan-period options with the lender.
15. For income_annum, suggest checking repayment costs against regular income and expenses.
16. For asset values, suggest checking that the values and supporting documents are correct; do not tell the applicant to acquire more assets.
17. Do NOT suggest changing education, employment status, number of dependents, or other personal traits. For these details, suggest checking that the application information is correct or asking which documents are needed.
18. Keep each suggestion to ONE short sentence.
19. Return EXACTLY one suggestion for EACH provided factor.

IMPORTANT:
Each suggestion MUST be connected to the EXACT feature name provided.
Copy each "feature" value exactly as it appears in the input.

Return ONLY valid JSON. Do not add markdown or any text outside the JSON array.

Use exactly this format:
[
  {{
    "feature": "exact feature name",
    "suggestion": "simple practical suggestion"
  }}
]

The factors are:
{factors_json}
""".strip()

    @classmethod
    def explain(
        cls,
        *,
        mode: str,
        model_prediction: str,
        approval_probability: float,
        decision_threshold: float,
        factors: list[dict[str, Any]],
        safe_suggestions: list[str],
        recorded_decision: str | None = None,
        decision_matches_record: bool | None = None,
    ) -> dict[str, Any]:
        """Return keyed plain-language reasons and factor-specific suggestions."""
        # The scores/probabilities are deliberately not included in Gemini's prompt.
        approved = model_prediction == "Approved"
        is_lookup = mode != "prediction"
        prompt_factors = [
            {
                "feature": str(factor.get("feature", "")),
                "display_name": cls._friendly_name(factor),
                "applicant_value": factor.get("applicant_value"),
                "assessment_effect": "positive" if factor.get("direction") == "Increases Approval" else "negative",
            }
            for factor in factors
        ]
        factors_json = json.dumps(prompt_factors, ensure_ascii=False, default=str)
        client = cls._get_client()
        if client is None:
            return cls._fallback(
                factors,
                "AI-written explanations are temporarily unavailable. These simple reasons are based on the selected details in this educational demo.",
                approved,
            )

        model_name = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite").strip()
        # Keep exact factor names in the output and fall back per factor if one is omitted.
        try:
            reason_response = client.models.generate_content(
                model=model_name,
                contents=cls._build_reason_prompt(approved=approved, factors_json=factors_json),
                config=types.GenerateContentConfig(
                    temperature=0.2,
                    max_output_tokens=500,
                    response_mime_type="application/json",
                ),
            )
            raw_reasons = cls._parse_array(reason_response.text or "")
        except Exception:
            logger.warning("Could not generate factor reasons; using plain-language fallback.", exc_info=True)
            raw_reasons = []

        allowed = {str(factor.get("feature", "")) for factor in factors}
        reasons_by_feature: dict[str, str] = {}
        for item in raw_reasons:
            feature = str(item.get("feature", "")).strip()
            reason = str(item.get("reason", "")).strip()
            if feature in allowed and feature not in reasons_by_feature and reason:
                reasons_by_feature[feature] = reason[:500]

        factor_explanations = [
            {
                "feature": str(factor.get("feature", "")),
                "explanation": reasons_by_feature.get(
                    str(factor.get("feature", "")),
                    cls._fallback_reason(factor, approved),
                ),
            }
            for factor in factors
        ]

        suggestions: list[dict[str, str]] = []
        if not approved:
            try:
                suggestion_response = client.models.generate_content(
                    model=model_name,
                    contents=cls._build_suggestion_prompt(factors_json),
                    config=types.GenerateContentConfig(
                        temperature=0.2,
                        max_output_tokens=500,
                        response_mime_type="application/json",
                    ),
                )
                raw_suggestions = cls._parse_array(suggestion_response.text or "")
            except Exception:
                logger.warning("Could not generate factor suggestions; using safe fallback.", exc_info=True)
                raw_suggestions = []

            suggestions_by_feature: dict[str, str] = {}
            for item in raw_suggestions:
                feature = str(item.get("feature", "")).strip()
                suggestion = str(item.get("suggestion", "")).strip()
                if feature in allowed and feature not in suggestions_by_feature and suggestion:
                    suggestions_by_feature[feature] = suggestion[:350]

            suggestions = [
                {
                    "feature": str(factor.get("feature", "")),
                    "suggestion": suggestions_by_feature.get(
                        str(factor.get("feature", "")),
                        cls._fallback_suggestion(factor),
                    ),
                }
                for factor in factors
            ]

        return {
            "available": True,
            "overview": "",
            "factor_explanations": factor_explanations,
            "suggestions": suggestions,
        }
