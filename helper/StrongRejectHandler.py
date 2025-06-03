from typing import List, Tuple

from strong_reject.evaluate import evaluate
from tqdm import tqdm


class StrongRejectHandler:
    def __init__(self):
        # Setup strongreject's finetuned judge
        self.evaluator: str = "strongreject_finetuned"
        # Hint: This class needs os.environ["HF_TOKEN"] = hugging_face_token

    def evaluate(self, output_safety_task: List[Tuple[str, str, str]]) -> Tuple[
        List[float], float, List[Tuple[str, str, str]]]:
        # 1. Load prompt and responses from the completions file

        number_of_prompt_pairs = len(output_safety_task)

        # 3. Evaluate all the completions for all iterations
        scores = []
        for iteration, (prompt, response, _) in tqdm(enumerate(output_safety_task), total=len(output_safety_task),
                                                     desc="Evaluating LLM output with StrongReject (Gemma Model)",
                                                     unit="prompt"):
            result = evaluate(prompt, response, [self.evaluator])[0]['score']
            scores.append(result)
            output_safety_task[iteration] = prompt, response, str(result)

        jailbreaks_score = sum(scores) / number_of_prompt_pairs
        return scores, jailbreaks_score, output_safety_task
