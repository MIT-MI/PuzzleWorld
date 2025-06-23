from data_loading import Sample
from modeling import EvalModel

PUZZLE_USER_PROMPT = """
Your task is to solve the following puzzle. The attached images are presented in the order
they are referenced in the text.

The puzzle's title is: {}
The puzzle's flavor text is: {}

---
Write out a step-by-step solution to the puzzle. At the end of your solution, write your
answer in the following format:
Answer: <answer>
"""

class Reasoner():
    def __init__(self, model):
        self.model=model
    
    def run(self, sample: Sample) -> str:
        raise NotImplementedError
    

class StandardReasoner(Reasoner):
    def __init__(self, model):
        super().__init__(model)
        self.prompt_template = PUZZLE_USER_PROMPT
        print("Loading model...")
        # self.model.load()
        print("Model loaded.")
        

    def run(self, sample: Sample, puzzle_content) -> str:
        reasoning_prompt = self.prompt_template.format(sample.title.rstrip(), sample.flavor_text)
        sample.prompt = reasoning_prompt
        response = "No Response Was Generated"

        # if type(self.model) == GPT4oModel:
        #     self.model.setTemp(0.0)
        self.model.setTemp(0.0)
        response = self.model.run(
            reasoning_prompt,
            puzzle_content
        )

        return response


def select_reasoner(name: str, model: EvalModel):
    if name == "standard":
        return StandardReasoner(model)
    
    raise KeyError(name)

def get_reasoner(reasoner: Reasoner) -> str:
    return str(type(reasoner))

