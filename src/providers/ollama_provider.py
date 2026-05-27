import logging

import ollama

from src.core.interfaces import LLMProvider

logger = logging.getLogger(__name__)


class OllamaProvider(LLMProvider):
    """Concrete implementation of LLMProvider using local Ollama instance."""

    def __init__(self, model_name: str = "llama3.1:8b", host: str = "http://localhost:11434"):
        """Initialize the Ollama provider.

        Args:
            model_name: The name of the local model to target (default: llama3.1:8b).
            host: The Ollama host address.
        """
        self.model_name = model_name
        self.client = ollama.Client(host=host)

    def generate_explanation(self, prompt: str, system_prompt: str | None = None) -> str:
        """Generate a response using the local Ollama instance.

        Args:
            prompt: The user query.
            system_prompt: Optional developer instruction.

        Returns:
            The generated response content.
        """
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            logger.info("Sending request to local Ollama model '%s'...", self.model_name)
            response = self.client.chat(model=self.model_name, messages=messages)
            content = response["message"]["content"]
            return content
        except Exception as e:
            logger.error("Failed to generate explanation with Ollama: %s", e, exc_info=True)
            return f"Error: Local LLM generation failed. Details: {e}"
