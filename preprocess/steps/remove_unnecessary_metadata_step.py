import re

from preprocess.preprocessing_step import PreprocessingStep


class RemoveUnnecessaryMetadata(PreprocessingStep):
    def __init__(self):
        super().__init__("remove_unnecessary_metadata")

    def process_text(self, lines) -> str:
        processed_lines = []

        for line in lines:
            # Remove date and time information
            line = re.sub(r'\(\d{1,2}:\d{2} [AP]M [A-Z]{3}\)', '', line)

            # Remove instant replay mentions
            line = re.sub(r'Instant Replay.*', '', line)

            # Only add non-empty lines after removal
            if line.strip():
                processed_lines.append(line.strip())

        return processed_lines
