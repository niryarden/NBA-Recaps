from preprocess.preprocessing_pipeline import PreprocessingPipeline
from preprocess.steps.add_special_tokens_step import AddSpecialTokens
from preprocess.steps.aggregate_similar_events_step import AggregateSimilarEvents
from preprocess.steps.normalize_team_names_step import NormalizeTeamNames
from preprocess.steps.remove_redundant_information_step import RemoveRedundantInformation
from preprocess.steps.remove_unnecessary_events_step import RemoveUnnecessaryEvents
from preprocess.steps.remove_unnecessary_metadata_step import RemoveUnnecessaryMetadata
from preprocess.steps.simplify_event_descriptions_step import SimplifyEventDescriptions
from preprocess.steps.simplify_time_representation_step import SimplifyTimeRepresentation
from preprocess.steps.standardize_event_format_step import StandardizeEventFormat
from preprocess.steps.tokenize_player_names_step import TokenizePlayerNames

class Preprocessor:
    def __init__(self):
        self.pipeline = PreprocessingPipeline()
        self._initialize_pipeline()

    def _initialize_pipeline(self):
        self.pipeline.add_step(RemoveUnnecessaryEvents())
        self.pipeline.add_step(AddSpecialTokens())
        self.pipeline.add_step(NormalizeTeamNames())
        self.pipeline.add_step(RemoveRedundantInformation())
        self.pipeline.add_step(RemoveUnnecessaryMetadata())
        self.pipeline.add_step(SimplifyEventDescriptions())
        self.pipeline.add_step(SimplifyTimeRepresentation())
        self.pipeline.add_step(StandardizeEventFormat())

    def preprocess(self, input_text: str) -> str:
        if not input_text:
            raise ValueError("Input text is empty or None")
        return self.pipeline.process(input_text)