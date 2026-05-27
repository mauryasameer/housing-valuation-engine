import logging

from src.core.interfaces import LLMProvider
from src.utils.security import clean_llm_output, detect_prompt_injection

logger = logging.getLogger(__name__)


class ExplanationService:
    """Orchestrates predictions, SHAP values, and LLM text generation,

    enforcing safety rules and input/output sanitization.
    """

    def __init__(self, llm_provider: LLMProvider):
        """Initialize the service.

        Args:
            llm_provider: Concrete instance implementing LLMProvider.
        """
        self.llm_provider = llm_provider

    def generate_narrative(self, predicted_price: float, shap_impacts: dict[str, float]) -> str:
        """Construct the prompt, send it to the LLM, and sanitize the response

        to create a clear, natural language explanation for a homeowner.

        Args:
            predicted_price: Model predicted sale price in dollars.
            shap_impacts: Dictionary mapping features to their marginal dollar impact.

        Returns:
            Sanitized narrative string.
        """
        # Group impacts into positive and negative drivers
        pos_drivers = []
        neg_drivers = []
        for feat, val in sorted(shap_impacts.items(), key=lambda item: abs(item[1]), reverse=True):
            if val > 0:
                pos_drivers.append(f"- {feat}: +${val:,.2f}")
            elif val < 0:
                neg_drivers.append(f"- {feat}: -${abs(val):,.2f}")

        # Extract top 5 positive and negative impacts
        pos_str = "\n".join(pos_drivers[:5]) if pos_drivers else "None"
        neg_str = "\n".join(neg_drivers[:5]) if neg_drivers else "None"

        # System prompt establishes the boundary instructions (Cybersecurity Prompt Hardening)
        system_prompt = (
            "You are a professional real estate valuation auditor. Your task is to explain a house's "
            "predicted market value and its primary value drivers (positive and negative) in a clear, "
            "natural language summary for a homeowner. Do NOT use technical jargon like 'SHAP', "
            "'log-attributions', or 'coefficients'. Explain the factors in plain English (e.g. 'extra living space "
            "added value', 'remodeling status reduced value'). Suggest 1-2 actionable upgrades to increase "
            "the home's value. Keep the response concise, under 200 words, and formatted as clean markdown."
        )

        # XML-fenced boundaries for inputs (Prompt Injection Protection)
        prompt = (
            f"Generate a valuation explanation report for the following property data:\n\n"
            f"<estimated_market_value>\n"
            f"${predicted_price:,.2f}\n"
            f"</estimated_market_value>\n\n"
            f"<positive_value_drivers>\n"
            f"{pos_str}\n"
            f"</positive_value_drivers>\n\n"
            f"<negative_value_drivers>\n"
            f"{neg_str}\n"
            f"</negative_value_drivers>\n\n"
            f"Provide a natural explanation report based on these factors."
        )

        # Cybersecurity Gate: Detect prompt injection
        if detect_prompt_injection(prompt) or detect_prompt_injection(system_prompt):
            logger.warning("Prompt injection signature detected. Blocking LLM generation request.")
            return "Security Block: The request was halted by the local AI security gate due to suspicious input signatures."

        logger.info("Requesting local LLM explanation narrative...")
        raw_output = self.llm_provider.generate_explanation(prompt, system_prompt=system_prompt)

        # Cybersecurity Gate: Output sanitization
        sanitized_output = clean_llm_output(raw_output)
        return sanitized_output
