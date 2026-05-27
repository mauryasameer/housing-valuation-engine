from abc import ABC, abstractmethod


class LLMProvider(ABC):
    """Abstract interface for LLM text generation providers."""

    @abstractmethod
    def generate_explanation(self, prompt: str, system_prompt: str | None = None) -> str:
        """Generate a text response given a prompt and optional system prompt.

        Args:
            prompt: The user prompt to send to the model.
            system_prompt: Optional instructions to shape the model's behavior.

        Returns:
            The model's generated text explanation.
        """
        ...
