import logging
from abc import abstractmethod, ABC
from typing import Callable

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


class PreprocessingStep(ABC):
    def __init__(self, name: str, enabled: bool = True):
        self.name = name
        self.enabled = enabled

    def apply(self, text: str) -> str:
        if not self.enabled:
            return text

        tokens_before = len("\\n".join(text))
        processed_text = self.process_text(text)
        tokens_after = len("\\n".join(processed_text))

        tokens_removed = tokens_before - tokens_after
        percentage_reduction = (tokens_removed / tokens_before * 100) if tokens_before > 0 else 0

        # logger.info(f"\nToken statistics for {self.name}:")
        # logger.info(f"Tokens before: {tokens_before}")
        # logger.info(f"Tokens after: {tokens_after}")
        # logger.info(f"Tokens removed: {tokens_removed}")
        # logger.info(f"Percentage reduction: {percentage_reduction:.2f}%")

        return processed_text

    @abstractmethod
    def process_text(self, text: str) -> str:
        pass