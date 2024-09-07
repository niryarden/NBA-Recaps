import re

from preprocess.preprocessing_step import PreprocessingStep


class AddSpecialTokens(PreprocessingStep):
    def __init__(self):
        super().__init__("add_special_tokens")

    def process_text(self, lines: str) -> str:
        processed_lines = []
        current_period = 0

        for line in lines:
            period_match = re.search(r'Period: (\d+)', line)
            if period_match:
                period = int(period_match.group(1))

                # Add end token for previous period if necessary
                if current_period > 0 and period != current_period:
                    processed_lines.append(f"<END_Q{current_period}>")

                # Add start token for new period
                if period != current_period:
                    processed_lines.append(f"<START_Q{period}>")

                    # Add halftime token after Q2
                    if period == 3:
                        processed_lines.append("<HALFTIME>")

                current_period = period

            processed_lines.append(line)

        # Add end token for the last period
        if current_period > 0:
            processed_lines.append(f"<END_Q{current_period}>")

        return processed_lines
