"""Applicant-friendly Gemini explanations grounded in locally computed evidence.

The model and feature-contribution calculations stay on the backend. Public responses
contain plain-language reasons and applicant values, never raw SHAP contribution values.
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
    """Turn the model's selected factors into simple explanations and safe suggestions."""

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
    def _fallback(
        factors: list[dict[str, Any]],
        safe_suggestions: list[str],
        reason: str,
    ) -> dict[str, Any]:
        """Provide plain-language fallback text; never expose technical contributions."""
        explanations = []
        for factor in factors:
            name = str(factor.get("feature", ""))
            display_name = str(factor.get("display_name") or name.replace("_", " ").title())
            direction = str(factor.get("direction", ""))
            if direction == "Increases Approval":
                text = f"{display_name} supported the demo's approval estimate."
            elif direction == "Decreases Approval":
                text = f"{display_name} moved the demo's estimate toward rejection."
            else:
                text = f"{display_name} was considered as part of the demo's estimate."
            explanations.append({"feature": name, "explanation": text})
        return {
            "available": False,
            "model": None,
            "overview": reason,
            "factor_explanations": explanations,
            "suggestions": list(safe_suggestions),
        }

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
        """Generate short, respectful explanations without allowing Gemini to alter a result."""
        client = cls._get_client()
        if client is None:
            return cls._fallback(
                factors,
                safe_suggestions,
                "A detailed AI-written explanation is unavailable right now. The result shown is an estimate from this educational demo, which uses synthetic data.",
            )

        is_lookup = mode != "prediction"
        prompt_data = {
            "workflow": "existing dataset record" if is_lookup else "new applicant details",
            "demo_estimate": model_prediction,
            "recorded_dataset_outcome": recorded_decision,
            "recorded_outcome_matches_estimate": decision_matches_record,
            "selected_details": [
                {
                    "feature": factor.get("feature"),
                    "plain_name": factor.get("display_name"),
                    "applicant_value": factor.get("applicant_value"),
                    "influence_on_demo_estimate": factor.get("direction"),
                }
                for factor in factors
            ],
            "suggestions_already_checked_for_safety": safe_suggestions,
        }
        prompt = f"""
You are a kind loan-explanation assistant. Write for a person who knows only basic English.
Use very simple English, common words, and short sentences.

Use ONLY the supplied JSON as evidence. The data is synthetic and is not a real lender's
policy. The outcome is an estimate from an educational demo, not a loan offer.

Return one valid JSON object with exactly these keys:
- "overview": a short, friendly summary in 2 or 3 sentences.
- "factor_explanations": an array with one object for EVERY selected detail. Each object has
  "feature" (copy the exact feature key) and "explanation" (one or two short sentences).
- "suggestions": an array of short strings.

Rules for explanations:
1. Explain each detail on its own. Do not combine or compare different details.
2. Say how the detail related to the demo's estimate, not that it definitely caused an outcome.
3. Use the supplied influence label to keep the explanation's direction correct.
4. Do not repeat the applicant's number or value in the explanation; it is displayed separately.
5. Do not mention SHAP, contribution values, feature importance, algorithms, or technical model terms.
6. Do not invent any facts, credit history, lender rules, cut-offs, or guarantees.
7. Avoid suggesting a person change personal traits such as education, employment status, or dependents.
8. For an existing record, the saved dataset outcome is only a label; it is not an official explanation
   from a bank. The factors describe the demo estimate, not the saved outcome's actual cause.
9. If the saved outcome differs from the demo estimate, clearly say they differ and that the
   explanation below applies to the demo estimate only.
10. Keep the overview clear and respectful.

Rules for suggestions:
1. Only rephrase the supplied safe suggestions. Do not add new advice.
2. Use basic English and make each suggestion one short sentence.
3. Do not promise that any action will lead to loan approval.
4. If the supplied suggestions list is empty, return an empty array.
5. Never mention SHAP or technical model details.

Evidence JSON:
{json.dumps(prompt_data, ensure_ascii=False, default=str)}
""".strip()

        model_name = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite").strip()
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.2,
                    max_output_tokens=800,
                    response_mime_type="application/json",
                ),
            )
            parsed = json.loads((response.text or "").strip())
            if not isinstance(parsed, dict):
                raise ValueError("Gemini response was not a JSON object.")

            allowed_features = {str(factor.get("feature", "")) for factor in factors}
            explanations: list[dict[str, str]] = []
            seen: set[str] = set()
            raw_explanations = parsed.get("factor_explanations", [])
            if isinstance(raw_explanations, list):
                for item in raw_explanations:
                    if not isinstance(item, dict):
                        continue
                    feature = str(item.get("feature", "")).strip()
                    explanation = str(item.get("explanation", "")).strip()
                    if feature in allowed_features and feature not in seen and explanation:
                        explanations.append(
                            {"feature": feature, "explanation": explanation[:500]}
                        )
                        seen.add(feature)

            # If Gemini omits an explanation, use a safe plain-language fallback for that detail.
            fallback_map = {
                item["feature"]: item["explanation"]
                for item in cls._fallback(factors, [], "")["factor_explanations"]
            }
            for factor in factors:
                name = str(factor.get("feature", ""))
                if name not in seen:
                    explanations.append(
                        {"feature": name, "explanation": fallback_map.get(name, "This detail was considered in the demo estimate.")}
                    )

            raw_suggestions = parsed.get("suggestions", [])
            generated_suggestions: list[str] = []
            if safe_suggestions and isinstance(raw_suggestions, list):
                generated_suggestions = [
                    str(value).strip()[:350]
                    for value in raw_suggestions
                    if isinstance(value, str) and value.strip()
                ][:len(safe_suggestions)]
            if safe_suggestions and not generated_suggestions:
                generated_suggestions = list(safe_suggestions)

            overview = str(parsed.get("overview", "")).strip()
            if not overview:
                raise ValueError("Gemini response did not contain an overview.")

            return {
                "available": True,
                "model": model_name,
                "overview": overview[:1200],
                "factor_explanations": explanations,
                "suggestions": generated_suggestions,
            }
        except Exception:
            logger.warning("Gemini explanation failed; using simple fallback text.", exc_info=True)
            return cls._fallback(
                factors,
                safe_suggestions,
                "A detailed AI-written explanation is unavailable right now. The result shown is an estimate from this educational demo, which uses synthetic data.",
            )
