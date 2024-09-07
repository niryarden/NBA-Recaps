from preprocess.preprocessing_step import PreprocessingStep


class RemoveUnnecessaryEvents(PreprocessingStep):
    def __init__(self):
        super().__init__("remove_unnecessary_events")

    def process_text(self, lines: str) -> str:
        filtered_lines = []
        for i, line in enumerate(lines):
            # Skip lines with MISS, REBOUND, SUB, or Timeout
            if any(event in line for event in ['MISS', 'REBOUND', 'SUB:', 'Timeout:']):
                continue

            # Check for fouls
            if 'FOUL' in line:
                # Keep the foul if it results in free throws
                if 'Free Throw' in line:
                    filtered_lines.append(line)
                elif 'S.FOUL' in line:
                    # Check next line for Free Throw
                    next_line = lines[i + 1] if i + 1 < len(lines) else ""
                    if 'Free Throw' in next_line:
                        filtered_lines.append(line)
                # Otherwise, skip the foul
                continue

            # Keep all other lines
            filtered_lines.append(line)

        return filtered_lines
