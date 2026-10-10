"""Responsible, conditional suggestions for the educational loan demo."""
from __future__ import annotations

from typing import List

from backend.schemas import LoanApplicationInput


class SuggestionService:
    @staticmethod
    def generate_suggestions(
        input_data: LoanApplicationInput,
        is_approved: bool,
        top_negative_factors: List[str],
        top_positive_factors: List[str],
    ) -> List[str]:
        """Return educational options, not promises or instructions to game a model.

        Demographic and other sensitive characteristics are not recommended for alteration.
        SHAP factors are accepted for future presentation but deterministic suggestions below
        are based on interpretable financial quantities only.
        """
        suggestions: List[str] = []
        income = max(float(input_data.income_annum), 1.0)
        loan_amount = float(input_data.loan_amount)
        dti_multiple = loan_amount / income
        total_assets = (
            max(0.0, float(input_data.residential_assets_value))
            + float(input_data.commercial_assets_value)
            + float(input_data.luxury_assets_value)
            + float(input_data.bank_asset_value)
        )
        asset_coverage = total_assets / loan_amount if loan_amount > 0 else 0.0

        if not is_approved:
            if input_data.cibil_score < 650:
                suggestions.append(
                    f"Credit report review: the supplied CIBIL score is {input_data.cibil_score}. Consider obtaining your official credit report, correcting verified errors, and maintaining on-time repayments. These steps may improve a future credit review but do not guarantee approval."
                )
            elif input_data.cibil_score < 720:
                suggestions.append(
                    f"Credit maintenance: the supplied CIBIL score is {input_data.cibil_score}. Review credit utilisation and repayment history, and avoid taking on debt that would be difficult to service. A different score does not guarantee a different lending outcome."
                )

            if dti_multiple > 3.0:
                illustrative_amount = int(income * 2.5)
                suggestions.append(
                    f"Affordability review: the requested amount is {dti_multiple:.1f} times the entered annual income. Compare the expected EMI against household expenses and existing debt. You could discuss a smaller amount (illustrative comparison only: ₹{illustrative_amount:,.0f}) or a different term with a qualified lender."
                )
            if asset_coverage < 1.0:
                suggestions.append(
                    f"Asset and liability review: the entered assets total about ₹{total_assets:,.0f}, compared with a requested loan of ₹{loan_amount:,.0f}. Verify asset valuations and any existing liabilities; collateral requirements vary by lender and product."
                )
            if float(input_data.bank_asset_value) < loan_amount * 0.15:
                suggestions.append(
                    "Cash-flow buffer: consider whether you can retain an emergency reserve for essential costs and repayments. Do not drain savings solely to improve a model score; compare the full cost and risks first."
                )
            if input_data.loan_term < 5 and dti_multiple > 2.0:
                suggestions.append(
                    "Repayment schedule: a short term can increase the monthly EMI. Ask a qualified lender to compare written repayment schedules for suitable terms, including total interest and fees."
                )
            if not suggestions:
                suggestions.append(
                    "Request a human review: the model output is only a synthetic-data estimate. Ask a qualified lender for the specific, verifiable factors relevant to its product and consider independent financial counselling before reapplying."
                )
        else:
            suggestions.append(
                "Maintain repayment capacity: keep an emergency reserve and make sure the expected EMI remains affordable if income or expenses change. The model output is not a loan offer or guarantee."
            )
            if dti_multiple > 3.5:
                suggestions.append(
                    f"Affordability caution: the requested amount is {dti_multiple:.1f} times the entered annual income. Review the EMI and total repayment against a realistic household budget before accepting any offer."
                )
            if float(input_data.bank_asset_value) > 0:
                suggestions.append(
                    "Preserve liquidity: confirm how much of the entered bank balance should remain available for emergencies instead of using it all for fees or upfront payments."
                )
            suggestions.append(
                "Compare written offers: review the annualised cost, interest type, processing fees, late-payment charges, prepayment terms, and repayment schedule before signing a loan agreement."
            )

        return suggestions
