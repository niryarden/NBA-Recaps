from transformers import pipeline


class QASimilarityMetric:
    def __init__(self, model_name='bert-base-uncased'):
        self.qa_model = pipeline('question-answering', model=model_name)

        self.questions = [
            "Who was the top scorer in the game?",
            "What was the final score of the game?",
            "Which team won the game?",
            "How many points did {team1} score in the game?",
            "How many points did {team2} score in the game?",
            "Who led {team1} in assists?",
            "Who led {team2} in assists?",
            "How many points did {team1} score in the first quarter?",
            "How many points did {team2} score in the first quarter?",
            "Who had the most rebounds in the game?",
            "How many three-pointers did {player1} make?",
            "How many turnovers did {team1} have?",
            "How many turnovers did {team2} have?",
            "What was the largest lead in the game?"
        ]
        self.game_scores = {}

    def answer_questions(self, recap_text, team1, team2, player1):
        personalized_questions = [q.format(team1=team1, team2=team2, player1=player1)
                                  for q in self.questions]

        answers = []
        for question in personalized_questions:
            result = self.qa_model(question=question, context=recap_text)
            answers.append(result['answer'])
 
        return answers

    def compute_metric(self, reference_text, generated_text, metadata, player1, game_id):
        team1 = metadata["home_team"]
        team2 = metadata["away_team"]
        ref_answers = self.answer_questions(reference_text, team1, team2, player1)
        gen_answers = self.answer_questions(generated_text, team1, team2, player1)

        correct_answers = sum(1 for ref, gen in zip(ref_answers, gen_answers) if ref == gen)
        total_questions = len(self.questions)

        score = correct_answers / total_questions
        self.game_scores[game_id] = score

        return score

    def get_top_5_generated_recaps(self):
        return sorted(self.game_scores, key=self.game_scores.get, reverse=True)[:5]

    def get_lowest_5_generated_recaps(self):
        return sorted(self.game_scores, key=self.game_scores.get)[:5]

    def get_average_score(self):
        return sum(self.game_scores.values()) / len(self.game_scores)
