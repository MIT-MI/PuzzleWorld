import asyncio
import nest_asyncio
nest_asyncio.apply()

from pydantic import BaseModel

from data_loading import Sample
from judge_utils import LLMJudge

GRADE_INSTRUCTION = """
Answer Equivalence Instructions:
Using the puzzle and the reference solution, grade the candidate solution as follows.

For every reasoning step of the reference solution, output True if the candidate solution both includes 
the step and achieves the same intermediate result of the step, otherwise False.
Explain why the candidate's solution did or did not get the reasoning step correct.
Do not add more steps than there are in the reference solution and evaluate every step 
in the reference solution. 
There is a exception in scoring for the last reasoning step. Identify the candidate output solution.
If the candiate output solution is the exact same as the reference solution answer of \"{puzzle_solution}\",
then output final step as true.

"""

PUZZLE_PROMPT = "The puzzle's title is {title}.\n Its flavor text is {flavor_text}."
QUERY_PROMPT = """
Explain your answer.
"""

class StepComparison(BaseModel):
    correct: bool
    explanation: str

    class Config:
        extra = "forbid"

class Evaluation(BaseModel):
    step_comparison: list[StepComparison]
    
    class Config:
        extra = "forbid"

class Scorer(BaseModel):
    def run(self, sample: Sample) -> float:
        raise NotImplementedError

class ExactScorer(Scorer):
    def run(self, sample: Sample) -> float:
        if sample.solution in sample.raw_output:
            return 1.0
        return 0.0

class GPTScorer(Scorer):
    def run(self, sample: Sample, folder_path) -> float:
        """
        Finds the score of a solution attempt.
        Score(reasoning_steps) is the index of the last correct reasoning step
        over total number of reasoning steps.

        params:
            sample: Sample, the data sample containing the annotated puzzle and solution attempt.
        returns:
            Returns the score of a solution attempt in the interval [0,1]. 
        """
        return self.run_with_explanation(sample, folder_path)[0]

    @classmethod
    def run_with_explanation(cls, sample: Sample, folder_path) -> tuple[float, str]:
        response, puzzle_data = cls.grade(folder_path, sample)
        candidate_output, puzzle_title, num_reasoning_steps = puzzle_data

        # Compute the ratio
        intermediate_score = 0
        explanation_lines = []
        for idx, comparison in enumerate(response.step_comparison, start=1):
            if comparison.correct:
                intermediate_score = idx
            explanation_lines.append(f"Step {idx}: {comparison.explanation}")
        score_ratio = intermediate_score / num_reasoning_steps
        return score_ratio, "\n".join(explanation_lines)

    @staticmethod
    async def check_answer_async(
        puzzle_prompt, puzzle_image, candidate_output,
        reasoning_steps, final_answer, content_path
    ) -> Evaluation:
        """
        Launch all stepwise evaluations concurrently.
        """
        puzzle_grade_instruction = GRADE_INSTRUCTION.format(puzzle_solution=final_answer)
        grader = LLMJudge(
            llm_str="gpt-4o",
            default_instructions=puzzle_grade_instruction
        )
        query = {"type": "text", "text": QUERY_PROMPT}

        tasks = [
            grader.parse_stepwise_completion(
                puzzle_prompt,
                puzzle_image,
                candidate_output,
                reasoning_steps,
                step_idx,
                content_path,
                query,
                response_format=StepComparison
            )
            for step_idx in range(len(reasoning_steps))
        ]

        raw_responses = await asyncio.gather(*tasks)

        evaluation = Evaluation(step_comparison=[])
        for raw in raw_responses:
            evaluation.step_comparison.append(raw.choices[0].message.parsed)

        return evaluation

    @staticmethod
    def grade(folder_path: str, data: Sample):
        candidate_output = data.raw_output
        reasoning_steps = data.reasoning
        final_answer = data.solution

        # Run the async checker once, gathering all step calls at once
        response = asyncio.run(
            GPTScorer.check_answer_async(
                PUZZLE_PROMPT.format(
                    title=data.title,
                    flavor_text=data.flavor_text
                ),
                data.puzzle_content,
                candidate_output,
                reasoning_steps,
                final_answer,
                folder_path
            )
        )

        data.raw_output = candidate_output
        return response, (candidate_output, data.title, len(reasoning_steps))