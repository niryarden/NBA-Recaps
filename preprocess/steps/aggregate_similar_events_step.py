import re

from preprocess.preprocessing_step import PreprocessingStep


class AggregateSimilarEvents(PreprocessingStep):
    def __init__(self):
        super().__init__("aggregate_similar_events")

    def process_text(self, lines) -> str:
        aggregated_lines = []
        current_player = None
        current_score = None
        consecutive_events = []

        for line in lines:
            player, event_type, points = self._parse_line(line)

            if player == current_player and event_type == 'score':
                consecutive_events.append((line, points))
            else:
                if consecutive_events:
                    aggregated_lines.append(self._aggregate_events(consecutive_events))
                consecutive_events = [(line, points)] if event_type == 'score' else []
                if event_type != 'score':
                    aggregated_lines.append(line)

            current_player = player
            if 'Score:' in line:
                current_score = re.search(r'Score: (\d+:\d+)', line).group(1)

        # Handle the last set of consecutive events
        if consecutive_events:
            aggregated_lines.append(self._aggregate_events(consecutive_events))

        return aggregated_lines

    def _parse_line(self, line):
        match = re.search(r'(\w+).*?\((\d+) PTS\)', line)
        if match:
            return match.group(1), 'score', int(match.group(2))
        else:
            player = line.split(',')[3].strip() if len(line.split(',')) > 3 else None
            return player, 'other', 0

    def _aggregate_events(self, events):
        if len(events) == 1:
            return events[0][0]

        base_line = events[0][0]
        total_points = sum(points for _, points in events)
        event_types = set(re.search(r'(\w+) \(\d+ PTS\)', event[0]).group(1) for event in events)
        event_description = f"{' and '.join(event_types)} ({total_points} PTS)"

        # Replace the original event description and points
        aggregated_line = re.sub(r'\w+ \(\d+ PTS\)', event_description, base_line)

        return aggregated_line
