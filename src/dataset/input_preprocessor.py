import re

from src.dataset.token_preprocessor import TokenPreprocessor


class InputPreprocessor:
    def __init__(self, args):
        self.args = args
        self.token_processor = TokenPreprocessor()

    def get_input(self, question, schema):
        return f'{self.args.prefix}: {question.strip()} {self.token_processor.preprocess(schema).strip()}'

    def get_label(self, label):
        return self.token_processor.preprocess(self.normalize(label))

    def normalize(self, query: str) -> str:
        def comma_fix(s):
            # Remove spaces in front of commas
            return s.replace(" , ", ", ")

        def white_space_fix(s):
            # Remove double and triple spaces
            return " ".join(s.split())

        def lower(s):
            # Convert everything except text between (single or double) quotation marks to lower case
            return re.sub(r"\b(?<!['\"])(\w+)(?!['\"])\b", lambda match: match.group(1).lower(), s)

        return comma_fix(white_space_fix(lower(query)))
