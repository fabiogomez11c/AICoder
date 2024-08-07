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


def create_response(client, model, messages, response_model, **kwargs):
    """Create a response using the instructor client"""
    return client.chat.completions.create(
        model=model,
        response_model=response_model,
        messages=messages,
        max_retries=CONFIG.get("llm").get("instructor_max_retries", 5),
        temperature=CONFIG.get("llm").get("temperature", 1),
        **kwargs
    )


def generate(
    prompt: BasePrompt, model: str, input_: Dict[str, str], **kwargs
) -> BaseModel:
    """
    Generate a response from the LLM chat, the result should be a valid pydantic model.
    """

    messages = [
        {
            "role": "system",
            "content": "You are an expert in content creation/writing with a highly focus on coherence.",
        },
        {"role": "user", "content": prompt.prompt_template.format(**input_)},
    ]

    result = create_response(
        client=create_openai_client(),
        model=model,
        messages=messages,
        response_model=prompt.pydantic_model,
        **kwargs
    )
    return result
