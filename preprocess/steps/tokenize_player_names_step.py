import re
from typing import Dict

from preprocess.preprocessing_step import PreprocessingStep


class TokenizePlayerNames(PreprocessingStep):
    def __init__(self):
        super().__init__("tokenize_player_names")
        self.player_lookup: Dict[str, str] = {}
        self.next_token_id = 1

    def process_text(self, lines) -> str:
        tokenized_lines = []

        for line in lines:
            tokenized_line = self._tokenize_line(line)
            tokenized_lines.append(tokenized_line)

        # Add the lookup table at the end of the processed text
        lookup_table = self._generate_lookup_table()
        return tokenized_lines + '\n\n' + lookup_table

    def _tokenize_line(self, line: str) -> str:
        # Regular expression to match player names (assuming names are two words)
        player_pattern = r'\b([A-Z][a-z]+ [A-Z][a-z]+)\b'

        def replace_name(match):
            full_name = match.group(1)
            if full_name not in self.player_lookup:
                token = f'P{self.next_token_id}'
                self.player_lookup[full_name] = token
                self.next_token_id += 1
            return self.player_lookup[full_name]

        return re.sub(player_pattern, replace_name, line)

    def _generate_lookup_table(self) -> str:
        table = "Player Name Lookup Table:\n"
        for full_name, token in self.player_lookup.items():
            table += f"{token}: {full_name}\n"
        return table