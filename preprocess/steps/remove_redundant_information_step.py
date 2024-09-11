import re

from preprocess.preprocessing_step import PreprocessingStep


class RemoveRedundantInformation(PreprocessingStep):
    def __init__(self):
        super().__init__("remove_redundant_information")
        self.current_score = None

    def process_text(self, lines) -> str:
        processed_lines = []

        for line in lines:
            processed_line = self._process_line(line)
            if processed_line:
                processed_lines.append(processed_line)

        return processed_lines

    def _process_line(self, line: str) -> str:
        # Remove "Event: " prefix
        line = line.replace("Event: ", "")

        # Extract score from the line
        score_match = re.search(r'Score: (\d+):(\d+)', line)
        if score_match:
            new_score = (int(score_match.group(1)), int(score_match.group(2)))

            # Check if the score has changed
            if self.current_score != new_score:
                self.current_score = new_score
                score_str = f"Score: {new_score[0]}:{new_score[1]}"
            else:
                # Remove score information if it hasn't changed
                line = re.sub(r'Score: \d+:\d+, ', '', line)

        return line
