import time
import pandas as pd
from openai import AzureOpenAI
import json


with open('.key/test-batch.json', 'r') as f:
    key = json.load(f)
    endpoint = key['endpoint']
    api_key = key['api_key']
    api_version = key['api_version']


client = AzureOpenAI(
  azure_endpoint = endpoint,
  api_key=api_key,
  api_version=api_version
)


def check_batch(batch_id: str):
    """
    Check the status of a batch job.
    """
    response = client.batches.retrieve(batch_id)
    return response
