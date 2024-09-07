import re
from datetime import datetime, timedelta

from preprocess.preprocessing_step import PreprocessingStep


class SimplifyTimeRepresentation(PreprocessingStep):
    def __init__(self):
        super().__init__("simplify_time_representation")

    def process_text(self, lines):
        simplified_lines = [self._simplify_line(line) for line in lines]
        return simplified_lines

    def _simplify_line(self, line: str) -> str:
        # Regular expression to match the time format
        time_pattern = r'Clock: (\d+):(\d+)\.(\d+)'

        def simplify_time(match):
            minutes = int(match.group(1))
            seconds = int(match.group(2))

            # Convert time to total seconds
            total_seconds = minutes * 60 + seconds

            # Round to nearest 30 seconds
            rounded_seconds = round(total_seconds / 30) * 30

            # Convert back to minutes and seconds
            new_minutes, new_seconds = divmod(rounded_seconds, 60)

            # Format the new time string
            return f'Clock: {new_minutes:02d}:{new_seconds:02d}'

        return re.sub(time_pattern, simplify_time, line)
