from typing import List

from preprocess.preprocessing_step import PreprocessingStep


class PreprocessingPipeline:
    def __init__(self):
        self.steps: List[PreprocessingStep] = []

    def add_step(self, step: PreprocessingStep):
        self.steps.append(step)

    def remove_step(self, step_name: str):
        self.steps = [step for step in self.steps if step.name != step_name]

    def enable_step(self, step_name: str):
        for step in self.steps:
            if step.name == step_name:
                step.enabled = True

    def disable_step(self, step_name: str):
        for step in self.steps:
            if step.name == step_name:
                step.enabled = False

    def process(self, text: str) -> str:
        lines = text.split("\n")
        for step in self.steps:
            lines = step.apply(lines)
        return "\n".join(lines)