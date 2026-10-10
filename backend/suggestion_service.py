from typing import List, Dict, Any
from backend.schemas import LoanApplicationInput

class SuggestionService:
    @staticmethod
    def generate_suggestions(
        input_data: LoanApplicationInput,
        is_approved: bool,
        top_negative_factors: List[str],
        top_positive_factors: List[str]
    ) -> TupleList:
        """
        Generates responsible, evidence-based, conditional financial educational suggestions.
        Never recommends altering demographic/protected attributes (dependents, education).
        Does not promise approval.
        """
        suggestions: List[str] = []
        income = input_data.income_annum if input_data.income_annum > 0 else 1.0
        dti = input_data.loan_amount / income
        total_assets = (
            max(0, input_data.residential_assets_value) +
            input_data.commercial_assets_value +
            input_data.luxury_assets_value +
            input_data.bank_asset_value
        )
        asset_coverage = total_assets / input_data.loan_amount if input_data.loan_amount > 0 else 0

        if not is_approved:
            # 1. Credit Score Analysis
            if input_data.cibil_score < 650:
                suggestions.append(
                    f"Credit Profile (CIBIL {input_data.cibil_score}): A credit score under 650 significantly reduces the model's approval confidence. "
                    "Consider checking your credit report from official bureaus for inaccuracies, disputing errors, and steadily establishing on-time payments "
                    "on existing debts before re-applying."
                )
            elif input_data.cibil_score < 720:
                suggestions.append(
                    f"Credit Maintenance (CIBIL {input_data.cibil_score}): While moderate, improving your score towards 750+ through consistent low credit utilization "
                    "and timely EMI repayments could strengthen future loan evaluations."
                )

            # 2. Debt-to-Income & Requested Amount
            if dti > 3.0:
                recommended_amount = int(income * 2.5)
                suggestions.append(
                    f"Loan Sizing & Debt Burden: The requested loan (₹{input_data.loan_amount:,.0f}) is {dti:.1f}x your annual income (₹{input_data.income_annum:,.0f}). "
                    f"Lenders generally seek lower debt-to-income multiples. Exploring a smaller loan request (e.g., around ₹{recommended_amount:,.0f}) "
                    "or extending the loan term may enhance affordability and repayment feasibility."
                )

            # 3. Asset & Reserve Backing
            if asset_coverage < 1.0:
                suggestions.append(
                    f"Collateral & Net Worth Backing: Total recorded assets (₹{total_assets:,.0f}) are less than the requested loan (₹{input_data.loan_amount:,.0f}). "
                    "Reviewing opportunities to offer verifiable collateral, additional co-signers with solid backing, or building higher savings reserves can mitigate lender risk."
                )

            # 4. Liquid Bank Balance
            if input_data.bank_asset_value < input_data.loan_amount * 0.15:
                suggestions.append(
                    f"Liquidity Buffer: Verifiable bank savings of ₹{input_data.bank_asset_value:,.0f} represent under 15% of the requested loan. "
                    "Maintaining a 3 to 6-month liquidity buffer can demonstrate readiness for unexpected economic shocks."
                )

            # 5. Loan Term consideration
            if input_data.loan_term < 5 and dti > 2.0:
                suggestions.append(
                    f"Repayment Horizon: A short loan term of {input_data.loan_term} years results in heavy monthly installment pressures relative to income. "
                    "Inquiring with a certified lending officer about a structured longer repayment tenure may lower periodic payments."
                )

            # Generic fallback if specific conditions are not triggered
            if not suggestions:
                suggestions.append(
                    "Comprehensive Underwriter Review: The model detected a combination of risk factors. "
                    "Consulting with a licensed loan officer or accredited financial counselor can help clarify institution-specific lending criteria and alternative financing options."
                )
        else:
            # Positive / Approved Guidance
            suggestions.append(
                f"Credit Stewardship: Your current CIBIL score of {input_data.cibil_score} is a strong positive pillar. "
                "Maintain this standing by avoiding multiple simultaneous credit inquiries and keeping credit card utilization below 30%."
            )
            if dti > 3.5:
                suggestions.append(
                    f"Affordability Caution: Although predicted approved, your loan amount is {dti:.1f}x your annual income. "
                    "Before signing a binding credit agreement, independently verify that the scheduled monthly EMI fits comfortably within your household budget."
                )
            if input_data.bank_asset_value > 0:
                suggestions.append(
                    f"Emergency Fund Preservation: With ₹{input_data.bank_asset_value:,.0f} in bank reserves, ensure your emergency fund remains liquid "
                    "rather than exhausting savings toward processing fees or upfront expenses."
                )
            suggestions.append(
                "Terms Comparison: Always request formal Key Fact Statements (KFS) from prospective lenders to compare Annual Percentage Rates (APR), "
                "foreclosure charges, and processing fees before committing."
            )

        return suggestions

TupleList = List[str]
