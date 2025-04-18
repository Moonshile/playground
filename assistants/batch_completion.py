import time
import pandas as pd
import json

from batch_infra import client, check_batch, endpoint


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

def convert_to_request(id: str, prompt: str, contents: list):
    return {
        'custom_id': id,
        'method': 'POST',
        'url': '/v1/chat/completions',
        'body': {
            'model': 'gpt-4o-2024-08-06-cosmo-Batch-test',
            'messages': [
                {'role': 'developer', 'content': prompt},
                {'role': 'user', 'content': contents}
            ],
            'temperature': 0.2,
        }
    }

def convert_to_requests(df: pd.DataFrame, prompt: str) -> list:
    """
    Convert DataFrame to a list of messages.
    """
    messages = []
    templ = {
        "custom_id": "request-1",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {"model": "gpt-3.5-turbo-0125", "messages": [{"role": "system", "content": "You are a helpful assistant."},{"role": "user", "content": "Hello world!"}],"max_tokens": 1000}}
    for index, row in df.iterrows():
        if not row['content'] and not row['extra']:
            continue
        contents = []
        if row['content'] and not pd.isna(row['content']):
            contents.append(convert_to_content(row['content']))
        if row['extra'] and not pd.isna(row['extra']):
            extra = json.loads(row['extra'])
            if 'images' in extra:
                for img in extra['images']:
                    contents.append(convert_to_content(img['url'], img=True))
        if contents:
            messages.append(convert_to_request(f'{index}-{row["userId"]}', prompt, contents))
    return messages

def save_batch_file(file_path: str, data: list):
    """
    Save the batch file.
    """
    with open(file_path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')


def upload_batch_file(file_path: str):
    """
    Upload the batch file.
    """
    with open(file_path, 'rb') as f:
        response = client.files.create(
            file=f,
            purpose="batch"
        )
    print("File uploaded successfully.")
    print(response)
    return response


def create_batch(file_id: str):
    """
    Create a batch job.
    """
    response = client.batches.create(
        input_file_id=file_id,
        completion_window='1h',
        endpoint=endpoint,
    )
    print("Batch job created successfully.")
    print(response)
    return response


def retrieve_batch_res(file_id: str, save_path: str):
    """
    Retrieve the results of a batch job.
    """
    response = client.files.content(file_id)
    with open(save_path, 'w') as f:
        f.write(response.text)
    print("Batch job results retrieved successfully.")


def generate(messages: list):
    response = client.chat.completions.create(
        model="gpt-4o-2024-08-06-cosmo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Does Azure OpenAI support customer managed keys?"},
            {"role": "assistant", "content": "Yes, customer managed keys are supported by Azure OpenAI."},
            {"role": "user", "content": "Do other Azure AI services support this too?"}
        ]
    )
    print(response.choices[0].message.content)


def main():
    import sys
    intent_filename: str = sys.argv[1]
    with open(intent_filename, 'r') as f:
        prompt = f.read()
    filename = sys.argv[2]
    # Read the CSV file
    df = read_csv(filename)
    # Convert DataFrame to a list of messages
    messages = convert_to_requests(df, prompt)
    # Save the batch file
    save_batch_file(filename + '.jsonl', messages)
    # Upload the batch file
    response = upload_batch_file(filename + '.jsonl')
    # Create a batch job
    batch_response = create_batch(response.id)
    # Check the status of the batch job
    batch_status = check_batch(batch_response.id)
    # Generate the response
    if batch_status.status == 'completed':
        # Retrieve the results of the batch job
        retrieve_batch_res(batch_status.output_file_id, filename + '_batch_result.jsonl')
        print("Batch job succeeded. Results saved to:", filename + '_batch_result.jsonl')
    else:
        print("Batch job failed.")
        print(batch_status)

if __name__ == "__main__":
    main()
