import asyncio
from PIL import Image
from typing import List, TypeVar, TypedDict
from openai import AsyncOpenAI, OpenAI
from openai.types import Completion
from pathlib import Path
# from openai.types.chat.parsed_chat_completion import ParsedChatCompletion
from tenacity import retry, stop_after_attempt, wait_random_exponential
from data_loading import convert_image_to_text, load_image
import os
import tiktoken

ResponseFormatT = TypeVar("ResponseFormatT")


class Message(TypedDict):
  role: str
  content: str

def create_message(role: str, content: str) -> Message:
  return {
    "role": role,
    "content": content
  }

class LLMJudge:
  max_image_size: int = 1024
  def __init__(self, llm_str: str = "gpt-4o", default_instructions: str | None = None, timeout: int | None = None):
    self.client = AsyncOpenAI(
      api_key=os.getenv("MIT_OPENAI_API_KEY"),
      timeout=timeout
    )
    self.llm_str = llm_str
    self.instructions = default_instructions

  def resize_image(self, image: Image) -> Image:
        h, w = image.size
        if h <= self.max_image_size and w <= self.max_image_size:
            return image

        factor = self.max_image_size / max(h, w)
        h = round(h * factor)
        w = round(w * factor)
        print(dict(old=image.size, resized=(h, w)))
        if image.mode == "RGBA":
            image = image.convert("RGB")
        image = image.resize((h, w), Image.LANCZOS)
        return image

  def make_puzzle_prompt(self, prompt: str, images: List[Image.Image] = None) -> List[dict]:
    inputs = [{"type": "text", "text": prompt}]
    if images is not None:
      for image in images:
        image_text = convert_image_to_text(self.resize_image(image))
        url = f"data:image/png;base64,{image_text}"
        inputs.append({"type": "image_url", "image_url": {"url": url}})
    return inputs
  
  def make_solution_prompt(self, prompt, reasoning_steps, content_path):
    inputs = [{"type": "text", "text": prompt}]
    for step_idx, step in enumerate(reasoning_steps):
      inputs.append({"type": "text", "text": f"{step_idx + 1}. {step.explanation}"})
      if step.figure is not None:
          figure_path = Path(content_path, step.figure)
          image = Image.open(figure_path)
          image_text = convert_image_to_text(self.resize_image(image))
          url = f"data:image/png;base64,{image_text}"
          inputs.append({"type": "image_url", "image_url": {"url": url}})
      if step_idx != len(reasoning_steps)-1:
        inputs.append({"type": "text", "text": f"Step {step_idx + 1} in candidate solution? (True/False) Explain."})
      else:
        inputs.append({"type": "text", "text": f"Is the correct final reference answer in the candidate output? (True/False) Explain."})
    return inputs
  
  def make_step_prompt(self, prompt, reasoning_steps, step_idx, content_path):
    inputs = [{"type": "text", "text": prompt}]
    step = reasoning_steps[step_idx]
    inputs.append({"type": "text", "text": f"{step_idx + 1}. {step.explanation}"})
    if step.figure is not None:
        figure_path = Path(content_path, step.figure)
        image = Image.open(figure_path)
        image_text = convert_image_to_text(self.resize_image(image))
        url = f"data:image/png;base64,{image_text}"
        inputs.append({"type": "image_url", "image_url": {"url": url}})
    if step_idx != len(reasoning_steps)-1:
      inputs.append({"type": "text", "text": f"Is step {step_idx + 1} in candidate solution? (True/False) Explain."})
    else:
      inputs.append({"type": "text", "text": f"Is the correct final reference answer in the candidate output? (True/False) Explain."})
    return inputs
  
  @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5)) 
  async def create_completion(self, puzzle_prompt, puzzle_image, candidate_output, reasoning_steps, content_path, query, **kwargs) -> Completion:
    instructions = kwargs.pop("instructions", self.instructions)

    if instructions is not None:
      system_message = create_message("system", instructions)
      full_puzzle_prompt = self.make_puzzle_prompt(puzzle_prompt, puzzle_image)
      candidate_output = ""
      full_candidate_prompt =[{"type": "text", "text":  f"Candidate Solution:\n{candidate_output}"}]
      full_solution_prompt = self.make_solution_prompt("Reference Solution:\n", reasoning_steps, content_path)

      user_prompt = {"role": "user", "content": full_puzzle_prompt + full_candidate_prompt + full_solution_prompt}
      messages = [system_message, user_prompt]
    
    response = await self.client.chat.completions.create(
      model=self.llm_str,
      messages=messages,
      **kwargs,
    )
    return response
  
  
  @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6)) 
  async def parse_completion(self, puzzle_prompt, puzzle_image, candidate_output, reasoning_steps, content_path, query, response_format: ResponseFormatT, **kwargs):
    instructions = kwargs.pop("instructions", self.instructions)

    if instructions is not None:
      system_message = create_message("system", instructions)
      full_puzzle_prompt = self.make_puzzle_prompt(puzzle_prompt, puzzle_image)
      full_candidate_prompt =[{"type": "text", "text":  f"Candidate Solution:\n{candidate_output}"}]
      full_solution_prompt = self.make_solution_prompt("Reference Solution:\n", reasoning_steps, content_path)

      user_prompt = {"role": "user", "content": full_puzzle_prompt + full_candidate_prompt + full_solution_prompt + [query]}
      messages = [system_message, user_prompt]
    response = await self.client.beta.chat.completions.parse(
      model=self.llm_str,
      messages=messages,
      response_format=response_format,
      **kwargs,
    )
    return response
  
  @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6)) 
  async def parse_stepwise_completion(self, puzzle_prompt, puzzle_image, candidate_output, reasoning_steps, step_idx, content_path, query, response_format: ResponseFormatT, **kwargs):
    instructions = kwargs.pop("instructions", self.instructions)

    if instructions is not None:
      system_message = create_message("system", instructions)
      full_puzzle_prompt = self.make_puzzle_prompt(puzzle_prompt, puzzle_image)
      full_candidate_prompt =[{"type": "text", "text":  f"Candidate Solution:\n{candidate_output}"}]
      partial_solution_prompt = self.make_step_prompt("Reference Solution:\n", reasoning_steps, step_idx, content_path)

      user_prompt = {"role": "user", "content": full_puzzle_prompt + full_candidate_prompt + partial_solution_prompt + [query]}
      messages = [system_message, user_prompt]
    response = await self.client.beta.chat.completions.parse(
      model=self.llm_str,
      messages=messages,
      response_format=response_format,
      **kwargs,
    )
    return response
  
  def count_tokens(self, messages: list[Message]) -> int:
    encoding = tiktoken.encoding_name_for_model(self.llm_str)
    tokens_per_message = 3
    tokens_per_name = 1
    num_tokens = 0
    for message in messages:
      num_tokens += tokens_per_message
      for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3
    return num_tokens
      

if __name__ == "__main__":
  llm = LLMJudge()
  message = create_message("user", "Hello, how are you?")
  completion = asyncio.run(llm.create_completion([message]))
  print(completion)