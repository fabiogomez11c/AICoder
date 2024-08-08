"""Script with the main"""

from typing import Dict, Any

from pydantic import BaseModel
from openai import OpenAI
import instructor
import yaml

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import asyncio

app = FastAPI()


class UserInput(BaseModel):
    message: str


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
        **kwargs,
    )


class AICoder(BaseModel):
    code: str


Response = instructor.Partial[AICoder]


async def invoke_ai_stream(message: str):
    """Generator function to stream AI responses"""
    client = create_openai_client()
    model = "gpt-4o-mini"  # Updated to a valid model name
    messages = [
        {
            "role": "system",
            "content": "You are an AI coding assistant, expert in good practices and good quality of code.",
        },
        {"role": "user", "content": message},
    ]
    response_model = Response

    response = create_response(client, model, messages, response_model, stream=True)
    for res in response:
        print(res)
        yield f"data: {res.code}\n\n"
        await asyncio.sleep(0.1)  # Small delay to control streaming rate


def invoke_ai(message: str) -> AICoder:
    """Function to get AI response without streaming"""
    client = create_openai_client()
    model = "gpt-4o-mini"  # Updated to a valid model name
    messages = [
        {
            "role": "system",
            "content": "You are an AI coding assistant, expert in good practices and good quality of code.",
        },
        {"role": "user", "content": message},
    ]
    response_model = AICoder

    response: AICoder = create_response(
        client, model, messages, response_model, stream=False
    )

    # Assuming the response is a single item
    return response


@app.post("/stream")
async def stream_ai_response(user_input: UserInput):
    """Endpoint to stream AI responses"""
    return StreamingResponse(
        invoke_ai_stream(user_input.message), media_type="text/event-stream"
    )


@app.post("/")
async def get_ai_response(user_input: UserInput):
    """Endpoint to get AI response"""
    ai_response: AICoder = invoke_ai(user_input.message)
    return {"response": ai_response.code}
