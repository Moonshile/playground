import pandas as pd
import os
import datetime
import json

from completion import generate, convert_to_content, parse_gen_res
from mysqlutils import assistants_conn


def process_one(row: pd.Series, prompt_message: dict):
    contents = []
    if row['content'] and not pd.isna(row['content']):
        contents.append(convert_to_content(row['content']))
    if row['extra'] and not pd.isna(row['extra']):
        extra = json.loads(row['extra'])
        if 'images' in extra:
            for img in extra['images']:
                contents.append(convert_to_content(img['url'], img=True))
    if not contents:
        return None
    resp = generate([prompt_message, {'role': 'user', 'content': contents}])
    res_text = parse_gen_res(resp)
    if not res_text:
        return None
    item = {
        'refMessageId': row['id'],
        'userId': row['userId'],
        'conversationId': row['conversationId'],
        'content': res_text,
        'status': 1,
        'type': 40,
    }
    return item


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
        item = process_one(row, prompt_message)
        if not item:
            print('[ERROR] No response from model for message {contents}')
            continue
        resps.append(row.to_dict())
        resps.append(item)
        if len(resps) > 0 and len(resps) % 10 == 0:
            print(f"[{datetime.datetime.now()}] Saving responses to {save_file}...")
            pd.DataFrame(resps).to_csv(save_file, index=False)


def generate_replys():
    prompt_filename = '.data/edu_problems.prompt.txt'
    with open(prompt_filename, 'r') as f:
        prompt = f.read()
    df = pd.read_csv('.data/edu_problems.csv', delimiter='\t')
    print(df.shape)
    process(df, prompt, '.data/edu_problems.res.csv')


def generate_one(idx: int):
    prompt_filename = '.data/edu_problems.prompt.txt'
    with open(prompt_filename, 'r') as f:
        prompt = f.read()
    df = pd.read_csv('.data/edu_problems.csv', delimiter='\t')
    print(df.shape)
    row = df.iloc[idx]
    print(row)
    prompt_message = {
        'role': 'developer',
        'content': prompt
    }
    item = process_one(row, prompt_message)
    if not item:
        print('[ERROR] No response from model for message {contents}')
        return
    print(item['content'])


def value_of_key(v):
    """
    Get the value of a key in a dictionary.
    """
    if v and not pd.isna(v):
        return v
    return None



def fill_into_db(df: pd.DataFrame):
    conn = assistants_conn()
    cursor = conn.cursor()
    for index, row in df.iterrows():
        print(f"[{datetime.datetime.now()}] Processing row {index+1}/{df.shape[0]}...")
        sql = 'insert into ai_assistant_message (refMessageId, userId, conversationId, content, status, type, extra) values (%s, %s, %s, %s, %s, %s, %s)'
        values = (
            value_of_key(row['refMessageId']),
            value_of_key(row['userId']),
            value_of_key(row['conversationId']),
            value_of_key(row['content']),
            value_of_key(row['status']),
            value_of_key(row['type']),
            value_of_key(row['extra']))
        cursor.execute(sql, values)
        conn.commit()
    cursor.close()
    conn.close()


if __name__ == "__main__":
    # generate_replys()
    generate_one(5)
    # df = pd.read_csv('.data/edu_problems.res.csv')
    # fill_into_db(df)

