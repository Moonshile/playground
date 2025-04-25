import os
import pandas as pd
from openai import AzureOpenAI
from openai.types.chat.chat_completion import ChatCompletion
import json
import datetime


with open('.key/cosmo.json', 'r') as f:
    key = json.load(f)
    endpoint = key['azure_endpoint']
    api_key = key['api_key']
    api_version = key['api_version']


client = AzureOpenAI(
  azure_endpoint = endpoint,
  api_key=api_key,
  api_version=api_version
)

def read_csv(file_path):
    """
    Read a CSV file and return a DataFrame.
    """
    df = pd.read_csv(file_path)
    return df

def convert_to_content(content: str, img = False):
    if img:
        return {
            'type': 'image_url',
            'image_url': {
                'url': content,
            }
        }
    return {
        'type': 'text',
        'text': content,
    }


def generate(messages: list, temperature: float = 0.2):
    response = client.chat.completions.create(
        model="gpt-4o-2024-08-06-cosmo",
        messages=messages,
        temperature=temperature,
    )
    return response


def parse_gen_res(response: ChatCompletion):
    if not response.choices:
        return None
    return response.choices[0].message.content


def analyze_response(response: ChatCompletion):
    """
    Analyze the responses from the model.
    """
    if not response.choices:
        return None
    try:
        intent = json.loads(response.choices[0].message.content)
        return intent
    except json.JSONDecodeError as e:
        print(f"Failed to parse response: {e}")
        raise e


def process(df: pd.DataFrame, prompt: str, save_file: str):
    prompt_message = {
        'role': 'developer',
        'content': prompt
    }
    resps = []
    skip = 0
    if os.path.exists(save_file):
        df_saved = pd.read_csv(save_file)
        skip = df_saved.shape[0]
    for index, row in df.iterrows():
        if index < skip:
            continue
        if not row['content'] and not row['extra']:
            continue
        print(f"[{datetime.datetime.now()}] Processing row {index+1}/{df.shape[0]}...")
        contents = []
        if row['content'] and not pd.isna(row['content']):
            contents.append(convert_to_content(row['content']))
        if row['extra'] and not pd.isna(row['extra']):
            extra = json.loads(row['extra'])
            if 'images' in extra:
                for img in extra['images']:
                    contents.append(convert_to_content(img['url'], img=True))
        if not contents:
            continue
        resp = generate([prompt_message, {'role': 'user', 'content': contents}])
        intent = analyze_response(resp)
        if intent:
            intent['userId'] = row['userId']
            intent['content'] = row['content']
            intent['extra'] = row['extra']
            intent['id'] = row['id']
            resps.append(intent)
            if len(resps) > 0 and len(resps) % 100 == 0:
                pd.DataFrame(resps).to_csv(save_file, index=False)
                print(f'saved {len(resps)} rows to {save_file}')
        else:
            print(f'failed to parse, {resp}')



def main():
    import sys
    intent_filename: str = sys.argv[1]
    with open(intent_filename, 'r') as f:
        prompt = f.read()
    filename = sys.argv[2]
    # Read the CSV file
    df = read_csv(filename)
    # Process the DataFrame
    output_filename = f'{filename}.res.csv'
    process(df, prompt, output_filename)

if __name__ == "__main__":
    main()
