import re

from preprocess.preprocessing_step import PreprocessingStep


class NormalizeTeamNames(PreprocessingStep):
    def __init__(self):
        super().__init__("normalize_team_names")
        self.team_abbreviations = {
            'Atlanta Hawks': 'ATL',
            'Boston Celtics': 'BOS',
            'Brooklyn Nets': 'BKN',
            'Charlotte Hornets': 'CHA',
            'Chicago Bulls': 'CHI',
            'Cleveland Cavaliers': 'CLE',
            'Dallas Mavericks': 'DAL',
            'Denver Nuggets': 'DEN',
            'Detroit Pistons': 'DET',
            'Golden State Warriors': 'GSW',
            'Houston Rockets': 'HOU',
            'Indiana Pacers': 'IND',
            'Los Angeles Clippers': 'LAC',
            'Los Angeles Lakers': 'LAL',
            'Memphis Grizzlies': 'MEM',
            'Miami Heat': 'MIA',
            'Milwaukee Bucks': 'MIL',
            'Minnesota Timberwolves': 'MIN',
            'New Orleans Pelicans': 'NOP',
            'New York Knicks': 'NYK',
            'Oklahoma City Thunder': 'OKC',
            'Orlando Magic': 'ORL',
            'Philadelphia 76ers': 'PHI',
            'Phoenix Suns': 'PHX',
            'Portland Trail Blazers': 'POR',
            'Sacramento Kings': 'SAC',
            'San Antonio Spurs': 'SAS',
            'Toronto Raptors': 'TOR',
            'Utah Jazz': 'UTA',
            'Washington Wizards': 'WAS'
        }

    def process_text(self, lines):
        text = "\n".join(lines)
        for full_name, abbreviation in self.team_abbreviations.items():
            # Replace full team name with abbreviation
            text = re.sub(r'\b' + re.escape(full_name) + r'\b', abbreviation, text)

            # Also replace team name without "The" if present
            if full_name.startswith('The '):
                text = re.sub(r'\b' + re.escape(full_name[4:]) + r'\b', abbreviation, text)

        return text.split("\n")
