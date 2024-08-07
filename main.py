"""Script with the main"""

from typing import Dict, Any

from pydantic import BaseModel
from openai import OpenAI
import instructor
import yaml


def read_yml_config(file_path: str) -> Dict[str, Any]:
    """
    Reads a YAML configuration file and returns its contents as a dictionary.

    :param file_path: Path to the YAML configuration file.
    :return: Dictionary containing the configuration.
    """
    with open(file_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def create_openai_client():
    """Create the instructor client for OpenAI"""
    config = read_yml_config("./keys.yaml")
    openai = config.get("openai")
    if openai:
        api_key = openai.get("key")
    else:
        raise ValueError("OpenAI api key failed")
    client = OpenAI(api_key=api_key)
    client = instructor.from_openai(client=client, mode=instructor.Mode.TOOLS)
    return client


def create_response(client, model, messages, response_model, **kwargs):
    """Create a response using the instructor client"""
    config = read_yml_config("./config.yaml")
    return client.chat.completions.create(
        model=model,
        response_model=response_model,
        messages=messages,
        max_retries=config.get("llm", {}).get("instructor_max_retries", 5),
        temperature=config.get("llm", {}).get("temperature", 1),
        **kwargs
    )


class Chat(BaseModel):
    ai_response: str


# Dummy example to create a response
if __name__ == "__main__":
    client = create_openai_client()
    model = "gpt-3.5-turbo"
    messages = [
        {"role": "user", "content": "Hello, how can I improve my coding skills?"}
    ]
    response_model = Chat

    response = create_response(client, model, messages, response_model)
    print(response.ai_response)


# def generate(
#     prompt: BasePrompt, model: str, input_: Dict[str, str], **kwargs
# ) -> BaseModel:
#     """
#     Generate a response from the LLM chat, the result should be a valid pydantic model.
#     """
#     pass
