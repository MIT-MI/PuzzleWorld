import asyncio
from typing import TypeVar, TypedDict
from openai import AsyncOpenAI, OpenAI
from openai.types import Completion
from openai.types.chat.parsed_chat_completion import ParsedChatCompletion
from tenacity import retry, stop_after_attempt, wait_random_exponential
import os
import json

ResponseFormatT = TypeVar("ResponseFormatT")


class Message(TypedDict):
  role: str
  content: str

def create_message(role: str, content: str) -> Message:
  return {
    "role": role,
    "content": content
  }

class LLM:
  def __init__(self, llm_str: str = "gpt-4o", default_instructions: str | None = None, timeout: int | None = None):
    self.client = AsyncOpenAI(
      api_key=os.getenv("MIT_OPENAI_API_KEY"),
      timeout=timeout
    )
    self.llm_str = llm_str
    self.instructions = default_instructions
  
  @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5)) 
  async def create_completion(self, messages: list[Message], **kwargs) -> Completion:
    instructions = kwargs.pop("instructions", self.instructions)

    if instructions is not None:
      system_message = create_message("system", instructions)
      messages = [system_message, *messages]
    
    response = await self.client.chat.completions.create(
      model=self.llm_str,
      messages=messages,
      **kwargs,
    )
    return response
  
  
  @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5)) 
  async def parse_completion(self, messages: list[Message], response_format: ResponseFormatT, **kwargs) -> ParsedChatCompletion[ResponseFormatT]:
    instructions = kwargs.pop("instructions", self.instructions)

    if instructions is not None:
      system_message = create_message("system", instructions)
      messages = [system_message, *messages]
    
    response = await self.client.beta.chat.completions.parse(
      model=self.llm_str,
      messages=messages,
      response_format=response_format,
      **kwargs,
    )
    return response

class BatchTask(TypedDict):
  task_id: str
  messages: list[Message]

class BatchLLM:
  def __init__(self, llm_str: str = "gpt-4o", default_instructions: str | None = None, timeout: int | None = None):
    self.client = OpenAI(
      api_key=os.getenv("MIT_OPENAI_API_KEY"),
    )
    self.llm_str = llm_str
    self.instructions = default_instructions
    self.endpoint = "/v1/chat/completions"
    self.batch_file = None
    self.batch_id = None
  
  def upload_batch_file(self, tasks: list[BatchTask], task_id: str, **kwargs):
    instructions = kwargs.pop("instructions", self.instructions)
    max_tokens = kwargs.pop("max_tokens", None)

    if instructions is not None:
      system_message = create_message("system", instructions)
      for task in tasks:
        task["messages"] = [system_message, *task["messages"]]
      
    method = "POST"
    
    batch_queries = []
    for task in tasks:
      batch_queries.append({
        "custom_id": task["task_id"],
        "method": method,
        "url": self.endpoint,
        "body": {
          "model": self.llm_str,
          "messages": task["messages"],
          "max_completion_tokens": max_tokens
        }
      })

    # save as jsonl file
    file_dest = f"batchinput_{task_id}.jsonl"
    with open(file_dest, "w", encoding="utf-8") as f:
      for query in batch_queries:
        f.write(json.dumps(query) + "\n")
        
    self.batch_file = self.client.files.create(
        file=open(file_dest, "rb"),
        purpose="batch"
    )
    
    return self.batch_file
  
  def create_batch_completion(self, description: str, batch_input_file_id:str = None):
    if batch_input_file_id is None:
      batch_input_file_id = self.batch_file.id
    response = self.client.batches.create(
      input_file_id=batch_input_file_id,
      endpoint=self.endpoint,
      completion_window="24h",
      metadata={
          "description": description
      }
    )
    self.batch_id = response.id
    return response
  
  def fetch_batch(self, batch_id: str = None):
    if batch_id is None:
      batch_id = self.batch_id
    response = self.client.batches.retrieve(batch_id)
    return response
  
  def fetch_completion(self, batch_id: str = None):
    batch = self.fetch_batch(batch_id)
    output_file_id = batch.output_file_id
    if output_file_id is None:
      return None
    file_response = self.client.files.content(output_file_id)
    # save text as jsonl file
    with open(f"batchoutput_{batch_id}.jsonl", "w") as f:
      f.write(file_response.text)
    
    return file_response
  
  def cancel_batch(self, batch_id: str = None):
    if batch_id is None:
      batch_id = self.batch_id
    response = self.client.batches.cancel(batch_id)
    return response

  

if __name__ == "__main__":
  llm = LLM()
  message = create_message("user", "Hello, how are you?")
  completion = asyncio.run(llm.create_completion([message]))
  print(completion)
