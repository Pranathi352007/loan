import os
import json
import numpy as np
import pandas as pd
import shap
from typing import List, Dict, Any, Tuple
from scipy.special import expit

from backend.schemas import LoanApplicationInput, ExplanationResponse, FeatureContribution
from backend.model_service import ModelService
from backend.suggestion_service import SuggestionService

FEATURE_METADATA = {
    'cibil_score': {
        'display_name': 'CIBIL Credit Score',
        'unit': 'points',
        'format': lambda v: f"{v}"
    },
    'loan_amount': {
        'display_name': 'Requested Loan Amount',
        'unit': 'INR',
        'format': lambda v: f"₹{v:,.0f}"
    },
    'income_annum': {
        'display_name': 'Annual Income',
        'unit': 'INR',
        'format': lambda v: f"₹{v:,.0f}"
    },
    'loan_term': {
        'display_name': 'Loan Term',
        'unit': 'years',
        'format': lambda v: f"{v} years"
    },
    'no_of_dependents': {
        'display_name': 'Number of Dependents',
        'unit': 'dependents',
        'format': lambda v: f"{v}"
    },
    'residential_assets_value': {
        'display_name': 'Residential Assets Value',
        'unit': 'INR',
        'format': lambda v: f"₹{v:,.0f}"
    },
    'commercial_assets_value': {
        'display_name': 'Commercial Assets Value',
        'unit': 'INR',
        'format': lambda v: f"₹{v:,.0f}"
    },
    'luxury_assets_value': {
        'display_name': 'Luxury Assets Value',
        'unit': 'INR',
        'format': lambda v: f"₹{v:,.0f}"
    },
    'bank_asset_value': {
        'display_name': 'Bank Asset Balance',
        'unit': 'INR',
        'format': lambda v: f"₹{v:,.0f}"
    },
    'education': {
        'display_name': 'Education Level',
        'unit': '',
        'format': lambda v: f"{v}"
    },
    'self_employed': {
        'display_name': 'Self-Employment Status',
        'unit': '',
        'format': lambda v: f"{v}"
    }
}

class ExplanationService:
    _instance = None

    def __init__(self):
        self.model_service = ModelService.get_instance()
        self.tree_explainer = None
        self._init_explainer()

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _init_explainer(self):
        clf = self.model_service.pipeline.named_steps['clf']
        self.tree_explainer = shap.TreeExplainer(clf)

    def explain(self, input_data: LoanApplicationInput) -> ExplanationResponse:
        pipeline = self.model_service.pipeline
        prep = pipeline.named_steps['prep']
        clf = pipeline.named_steps['clf']

        df = self.model_service.prepare_dataframe(input_data)
        
        # Transform the applicant using the exact preprocessing pipeline used for training.
        X_trans = prep.transform(df)

        # SHAP versions differ in whether a classifier output axis is retained.
        # Always extract one contribution per transformed feature for the approval output.
        shap_explanation = self.tree_explainer(X_trans)
        shap_values = np.asarray(shap_explanation.values)
        output_index = 0
        if shap_values.ndim == 3:
            output_index = 1 if shap_values.shape[-1] > 1 else 0
            raw_shap_values = shap_values[0, :, output_index].reshape(-1)
        elif shap_values.ndim == 2:
            raw_shap_values = shap_values[0, :].reshape(-1)
        else:
            raise ValueError(
                f"Unexpected SHAP value shape {shap_values.shape}; expected a sample-by-feature array."
            )

        if len(raw_shap_values) != X_trans.shape[1]:
            raise ValueError(
                "SHAP output feature count does not match the transformed model input."
            )

        # TreeExplainer's expected value is on the raw model-output scale.
        # For a multi-output result, choose class 1 (Approved) when available.
        base_values = np.asarray(self.tree_explainer.expected_value)
        if base_values.size > 1:
            selected_index = min(output_index, base_values.size - 1)
            base_val = float(base_values.reshape(-1)[selected_index])
        else:
            base_val = float(base_values.reshape(-1)[0])

        base_prob = float(expit(base_val))

        # Model predicted probability
        probs = pipeline.predict_proba(df)[0]
        approval_prob = float(probs[1])
        threshold = float(self.model_service.metadata.get('selected_threshold', 0.50))
        is_approved = bool(approval_prob >= threshold)
        predicted_status = "Approved" if is_approved else "Rejected"

        # Map transformed feature contributions back to original feature names
        all_features = self.model_service.metadata['feature_schema']['features']
        num_cols = self.model_service.metadata['feature_schema']['numerical_features']
        cat_cols = self.model_service.metadata['feature_schema']['categorical_features']
        
        cat_encoder = prep.named_transformers_['cat'].named_steps['encoder']
        cat_feature_names = list(cat_encoder.get_feature_names_out(cat_cols))
        transformed_names = num_cols + cat_feature_names

        # Map transformed features to original
        orig_contributions: Dict[str, float] = {feat: 0.0 for feat in all_features}
        for name, val in zip(transformed_names, raw_shap_values):
            matched = None
            for orig in all_features:
                if name == orig or name.startswith(orig + '_'):
                    matched = orig
                    break
            if matched:
                orig_contributions[matched] += float(val)

        # Rank features by absolute magnitude
        sorted_feats = sorted(orig_contributions.items(), key=lambda x: abs(x[1]), reverse=True)

        feature_contributions_list: List[FeatureContribution] = []
        positive_factors: List[str] = []
        negative_factors: List[str] = []

        data_dict = input_data.model_dump()

        for rank, (feat_name, shap_val) in enumerate(sorted_feats):
            meta = FEATURE_METADATA.get(feat_name, {
                'display_name': feat_name.replace('_', ' ').title(),
                'format': lambda v: str(v)
            })
            display_name = meta['display_name']
            raw_val = data_dict.get(feat_name, None)
            formatted_val = meta['format'](raw_val)

            direction = "Increases Approval" if shap_val >= 0 else "Decreases Approval"

            if rank == 0:
                strength = "Dominant"
            elif rank in [1, 2]:
                strength = "High"
            elif rank in [3, 4, 5]:
                strength = "Moderate"
            else:
                strength = "Minor"

            # Contextual feature explanation
            explanation_text = self._generate_feature_explanation(feat_name, display_name, raw_val, shap_val, data_dict)

            fc = FeatureContribution(
                feature=feat_name,
                display_name=display_name,
                applicant_value=formatted_val,
                shap_value=round(shap_val, 4),
                direction=direction,
                impact_strength=strength,
                explanation=explanation_text
            )
            feature_contributions_list.append(fc)

            factor_desc = f"{display_name} ({formatted_val}): {direction} ({'+' if shap_val>0 else ''}{shap_val:.3f} log-odds contribution)"
            if shap_val > 0:
                positive_factors.append(factor_desc)
            elif shap_val < 0:
                negative_factors.append(factor_desc)

        # Synthesize top reasons from actual computed forces
        reasons = self._synthesize_reasons(sorted_feats, data_dict, is_approved, approval_prob)

        # Generate responsible, educational suggestions
        suggestions = SuggestionService.generate_suggestions(
            input_data=input_data,
            is_approved=is_approved,
            top_negative_factors=[f[0] for f in sorted_feats if f[1] < 0][:3],
            top_positive_factors=[f[0] for f in sorted_feats if f[1] > 0][:3]
        )

        most_influential = self.model_service.metadata.get('most_influential_feature', 'cibil_score')
        top_curr = sorted_feats[0][0]
        model_dependence_note = (
            f"Global analysis identifies '{most_influential}' as the model's most influential feature overall. "
            f"For this applicant profile, the primary driving force was '{top_curr}' "
            f"({'+' if sorted_feats[0][1]>0 else ''}{sorted_feats[0][1]:.3f} log-odds impact)."
        )

        disclaimer = (
            "EXPLAINABILITY DISCLAIMER: The contributions shown are mathematically exact SHAP (TreeExplainer) values "
            "explaining the internal mechanics of the trained Gradient Boosting model on this simulated dataset. "
            "They do not depict internal deliberation or policy criteria of any actual commercial bank."
        )

        return ExplanationResponse(
            predicted_status=predicted_status,
            is_approved=is_approved,
            approval_probability=round(approval_prob, 4),
            base_probability=round(base_prob, 4),
            decision_threshold=threshold,
            feature_contributions=feature_contributions_list,
            top_positive_factors=positive_factors[:3],
            top_negative_factors=negative_factors[:3],
            reasons=reasons,
            suggestions=suggestions,
            model_dependence_note=model_dependence_note,
            disclaimer=disclaimer
        )

    def _generate_feature_explanation(self, feat: str, display: str, val: Any, shap_val: float, all_vals: Dict[str, Any]) -> str:
        sign = "positive" if shap_val >= 0 else "negative"
        direction_phrase = "increased the model's predicted approval likelihood" if shap_val >= 0 else "reduced the model's predicted approval likelihood"

        contribution_text = f"SHAP contribution {shap_val:+.3f} on the model's raw output scale {direction_phrase}."

        if feat == 'cibil_score':
            return f"CIBIL score of {val} had a learned association with the model output; {contribution_text}"

        elif feat == 'loan_amount':
            income = float(all_vals.get('income_annum', 1) or 1)
            ratio = float(val) / income if income > 0 else 0
            return f"Requested loan of ₹{val:,.0f} is {ratio:.2f} times the entered annual income; {contribution_text}"

        elif feat == 'income_annum':
            return f"Entered annual income of ₹{val:,.0f}; {contribution_text}"

        elif feat == 'loan_term':
            return f"Entered loan duration of {val} years; {contribution_text}"

        elif feat == 'no_of_dependents':
            return f"Entered number of dependents ({val}); {contribution_text}"

        elif 'assets_value' in feat or feat == 'bank_asset_value':
            return (
                f"Recorded {display.lower()} of ₹{val:,.0f}; {contribution_text} "
                "The dataset does not verify that assets are pledged collateral."
            )

        elif feat == 'education':
            return f"Education category ('{val}') followed a learned pattern in this synthetic dataset; {contribution_text}"

        elif feat == 'self_employed':
            return f"Self-employment category ('{val}') followed a learned pattern in this synthetic dataset; {contribution_text}"

        return f"{display} with value {val}; {contribution_text}"

    def _synthesize_reasons(self, sorted_feats: List[Tuple[str, float]], vals: Dict[str, Any], is_approved: bool, prob: float) -> List[str]:
        reasons = []
        top_pos = [f for f in sorted_feats if f[1] > 0][:2]
        top_neg = [f for f in sorted_feats if f[1] < 0][:2]

        if is_approved:
            for feat, val in top_pos:
                meta = FEATURE_METADATA.get(feat, {'display_name': feat})
                reasons.append(
                    f"Strong Positive Driver: {meta['display_name']} contributed favorably (+{val:.3f} log-odds) toward an approval likelihood of {prob*100:.1f}%."
                )
            if top_neg:
                feat, val = top_neg[0]
                meta = FEATURE_METADATA.get(feat, {'display_name': feat})
                reasons.append(
                    f"Primary Risk Drag: Despite overall approval, {meta['display_name']} was the largest countervailing negative factor ({val:.3f} log-odds)."
                )
        else:
            for feat, val in top_neg:
                meta = FEATURE_METADATA.get(feat, {'display_name': feat})
                reasons.append(
                    f"Primary Denial Drag: {meta['display_name']} exerted significant downward pressure ({val:.3f} log-odds) on approval probability."
                )
            if top_pos:
                feat, val = top_pos[0]
                meta = FEATURE_METADATA.get(feat, {'display_name': feat})
                reasons.append(
                    f"Compensating Strength: {meta['display_name']} provided positive support (+{val:.3f} log-odds), but was insufficient to overcome the primary negative factors."
                )

        return reasons
