import re
from typing import List, Tuple

from preprocess.preprocessing_step import PreprocessingStep


class SimplifyEventDescriptions(PreprocessingStep):
    def __init__(self):
        super().__init__("simplify_event_descriptions")

    def process_text(self, lines) -> str:
        simplified_lines = []
        i = 0
        while i < len(lines):
            line = lines[i]
            if "Jump Shot" in line or "Layup" in line or "Dunk" in line:
                simplified_lines.append(self._simplify_shot(line))
            elif "Free Throw" in line:
                simplified_line, i = self._combine_free_throws(lines, i)
                simplified_lines.append(simplified_line)
            elif "Turnover" in line:
                simplified_lines.append(self._simplify_turnover(line))
            else:
                simplified_lines.append(line)
            i += 1
        return simplified_lines

    def _simplify_shot(self, line: str) -> str:
        parts = line.split(', ')
        event = parts[-1]
        match = re.search(r'(\w+) (\d+\' )?(.*) \((\d+) PTS\)', event)
        if match:
            player, _, shot_type, points = match.groups()
            shot_type = '3PT' if '3PT' in shot_type else shot_type.split()[0]
            return f"{', '.join(parts[:-1])}, {player} {shot_type} ({points} PTS)"
        return line

    def _combine_free_throws(self, lines: List[str], start_index: int) -> Tuple[str, int]:
        current_line = lines[start_index]
        parts = current_line.split(', ')
        event = parts[-1]
        match = re.search(r'(\w+) Free Throw (\d) of (\d) \((\d+) PTS\)', event)
        if not match:
            return current_line, start_index

        player, _, total_throws, points = match.groups()
        made_throws = 1
        total_points = int(points)

        for i in range(start_index + 1, len(lines)):
            next_line = lines[i]
            if player not in next_line or "Free Throw" not in next_line:
                break
            next_match = re.search(r'Free Throw \d of \d \((\d+) PTS\)', next_line)
            if next_match:
                made_throws += 1
                total_points += int(next_match.group(1))
            else:
                break

        simplified_event = f"{player} Free Throw {made_throws}/{total_throws} ({total_points} PTS)"
        return f"{', '.join(parts[:-1])}, {simplified_event}", start_index + made_throws - 1

    def _simplify_turnover(self, line: str) -> str:
        parts = line.split(', ')
        event = parts[-1]
        match = re.search(r'(\w+) (.* )?Turnover', event)
        if match:
            player = match.group(1)
            return f"{', '.join(parts[:-1])}, {player} Turnover"
        return line
