import urllib.request
import json
import os
import ssl
import requests
import time
import pandas as pd

from openai import OpenAI
import anthropic
# from google.cloud import aiplatform
# import vertexai
# from vertexai.preview.language_models import TextGenerationModel, ChatModel

'''
Close-source models
Included:
- GPT-4-turbo, 
- GPT-3.5-turbo, 
- Claude 2, 
- Claude 2 Instant, and 
- PaLM 2 (currently commented out)
TODO:
- Gemini
'''

## Load secrets
## search the file in different levels
SECRET_FILE = '../secrets.txt'

if not os.path.exists(SECRET_FILE):
    SECRET_FILE = 'secrets.txt'
    if not os.path.exists(SECRET_FILE):
        SECRET_FILE = '../../secrets.txt'


with open(SECRET_FILE) as f:
    lines = f.readlines()
    for line in lines:
        if line.split(',')[0].strip() == "open_ai_key":
            open_ai_key = line.split(',')[1].strip()
        elif line.split(',')[0].strip() == "anthropic_key":
            anthropic_key = line.split(',')[1].strip()
        elif line.split(',')[0].strip() == "palm_project_id":
            palm_project_id = line.split(',')[1].strip()

openai_client = OpenAI(api_key=open_ai_key)
claude_client = anthropic.Anthropic(api_key=anthropic_key)
# vertexai.init(project = palm_project_id, location="us-central1")
# chat_model = ChatModel.from_pretrained("chat-bison@001")

### Run Models

# GPT models (GPT-3.5 and GPT-4)
def run_gpt(text_prompt, max_tokens_to_sample: int = 2000, temperature: float = 0, client=openai_client, model = "gpt-3.5-turbo"):
    # use gpt-3.5-turbo unless specify gpt-4
    response = client.chat.completions.create(
      model = model, 
      messages=[
        {"role": "user", "content": text_prompt},
      ],
      temperature=temperature,
      max_tokens=max_tokens_to_sample
    )
    return response.choices[0].message.content


# Claude 2 models (Claude 2 and Claude 2 Instant)
def run_claude(text_prompt, max_tokens_to_sample: int = 3000, temperature: float = 0, client=claude_client, model = "claude-instant-1.2"):
    # use claude-instant unless specify claude-2
    prompt = f"{anthropic.HUMAN_PROMPT} {text_prompt}{anthropic.AI_PROMPT}"
    resp = client.completions.create(
        prompt=prompt,
        stop_sequences=[anthropic.HUMAN_PROMPT],
        model=model, # model="claude-v1.3-100k",
        max_tokens_to_sample=max_tokens_to_sample,
        temperature=temperature,
    ).completion
    return resp

# PaLM 2 model

# def run_palm(text_prompt, max_tokens_to_sample: int = 1000, temperature: float = 0, model = "chat-bison@001"):
#     """Use Google PaLM chat model"""
#     parameters = {
#         "temperature": temperature, 
#         "max_output_tokens": max_tokens_to_sample
#     }
#     chat = chat_model.start_chat()
#     response = chat.send_message(text_prompt, **parameters)
#     response = response.text
#     time.sleep(2)
#     return response


