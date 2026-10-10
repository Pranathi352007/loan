from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, field_validator

class LoanApplicationInput(BaseModel):
    no_of_dependents: int = Field(..., ge=0, le=10, description="Number of dependents", examples=[2])
    education: str = Field(..., description="Education level: 'Graduate' or 'Not Graduate'", examples=["Graduate"])
    self_employed: str = Field(..., description="Self-employment status: 'Yes' or 'No'", examples=["No"])
    income_annum: float = Field(..., gt=0, description="Annual income in INR; must be greater than zero", examples=[6000000])
    loan_amount: float = Field(..., gt=0, description="Requested loan amount in INR", examples=[15000000])
    loan_term: int = Field(..., ge=1, le=30, description="Loan term in years", examples=[10])
    cibil_score: int = Field(..., ge=300, le=900, description="CIBIL credit score between 300 and 900", examples=[750])
    residential_assets_value: float = Field(..., ge=0, description="Value of residential assets in INR", examples=[12000000])
    commercial_assets_value: float = Field(..., ge=0, description="Value of commercial assets in INR", examples=[4000000])
    luxury_assets_value: float = Field(..., ge=0, description="Value of luxury assets in INR", examples=[18000000])
    bank_asset_value: float = Field(..., ge=0, description="Bank asset balance/savings in INR", examples=[8000000])

    @field_validator('education')
    @classmethod
    def validate_education(cls, v: str) -> str:
        clean = v.strip().title()
        if clean not in ["Graduate", "Not Graduate"]:
            raise ValueError("Education must be either 'Graduate' or 'Not Graduate'")
        return clean

    @field_validator('self_employed')
    @classmethod
    def validate_self_employed(cls, v: str) -> str:
        clean = v.strip().capitalize()
        if clean not in ["Yes", "No"]:
            raise ValueError("Self-employed must be either 'Yes' or 'No'")
        return clean


class PredictionResponse(BaseModel):
    predicted_status: str
    is_approved: bool
    approval_probability: float
    rejection_probability: float
    decision_threshold: float
    confidence_level: str
    debt_to_income_ratio: float
    summary: str
    disclaimer: str


class FeatureContribution(BaseModel):
    feature: str
    display_name: str
    applicant_value: Any
    shap_value: float
    direction: str  # "Increases Approval" or "Decreases Approval"
    impact_strength: str  # "Dominant", "High", "Moderate", "Minor"
    explanation: str


class ExplanationResponse(BaseModel):
    predicted_status: str
    is_approved: bool
    approval_probability: float
    base_probability: float
    decision_threshold: float
    feature_contributions: List[FeatureContribution]
    top_positive_factors: List[str]
    top_negative_factors: List[str]
    reasons: List[str]
    suggestions: List[str]
    model_dependence_note: str
    disclaimer: str


class ModelInfoResponse(BaseModel):
    model_name: str
    pipeline_steps: List[str]
    features: List[str]
    numerical_features: List[str]
    categorical_features: List[str]
    categorical_options: Dict[str, List[Any]]
    target_mapping: Dict[str, str]
    selected_threshold: float
    metrics_summary: Dict[str, float]
    most_influential_feature: str
    synthetic_disclaimer: str
