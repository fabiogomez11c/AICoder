"""Script with the main"""

from typing import Dict, Any

from pydantic import BaseModel
from openai import OpenAI
import instructor
import yaml

from fastapi import FastAPI

app = FastAPI()


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
    # The code attribute represents the programming code as a string.
    code: str


def invoke_ai(message: str):
    """Script to invoke the AI"""
    client = create_openai_client()
    model = "gpt-4o-mini"
    messages = [
        {
            "role": "system",
            "content": "You are an AI coding assistant, expert in good practices and good quality of code.",
        },
        {"role": "user", "content": message},
    ]
    response_model = Chat

    response = create_response(client, model, messages, response_model)
    return response.code


class UserInput(BaseModel):
    message: str


@app.post("/")
async def get_ai_response(user_input: UserInput):
    """Endpoint to get AI response"""
    ai_response = invoke_ai(user_input.message)
    return {"response": ai_response}
