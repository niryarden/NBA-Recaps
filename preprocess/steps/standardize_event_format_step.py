import re

from preprocess.preprocessing_step import PreprocessingStep


class StandardizeEventFormat(PreprocessingStep):
    def __init__(self):
        super().__init__("standardize_event_format")

    def process_text(self, lines) -> str:
        standardized_lines = [self._standardize_line(line) for line in lines]
        return standardized_lines

    def _standardize_line(self, line: str) -> str:
        # Regular expression to match the current format
        pattern = r'Period: (\d+), Clock: (\d+:\d+), Score: (\d+):(\d+), Event: (.+)'
        match = re.match(pattern, line)

        if not match:
            return line  # Return the original line if it doesn't match the expected format

        period, time, score1, score2, event = match.groups()
        score = f"{score1}-{score2}"

        # Extract player and points information
        player, points = self._extract_player_and_points(event)

        # Construct the standardized format
        return f"{period},{time},{score},{player},{event},{points}"

    def _extract_player_and_points(self, event: str) -> tuple:
        # Extract player name (assuming it's the first word in the event)
        player = event.split()[0]

        # Extract points (if available)
        points_match = re.search(r'\((\d+) PTS\)', event)
        points = points_match.group(1) if points_match else "0"

        return player, points