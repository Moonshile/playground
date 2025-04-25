import pandas as pd
import json


raw = pd.read_csv('../.data/20250418.csv.res.csv')

edu = raw[raw['intent']=='EDU_ASSESSMENT_PROBLEM']
edu.to_csv('../.data/20250418.edu.csv')

feat = raw[raw['intent']=='APP_FEATURE']
feat.to_csv('../.data/20250418.appfeat.csv')


def to_txt(df: pd.DataFrame, fn: str):
    images = {}
    texts = {}
    for index, row in df.iterrows():
        if not pd.isna(row['extra']):
            extra = json.loads(row['extra'])
            if 'images' in extra:
                for img in extra['images']:
                    images[img['url']] = row['content']
        elif not pd.isna(row['content']):
            texts[row['content']] = 1
    res = [(str(v) + ": " if v and not pd.isna(v) else '') + k for k, v in images.items()]
    res += [k for k, v in texts.items()]
    text_content = "\n\n---\n\n".join(res)
    with open(fn, 'w') as f:
        f.write(text_content)


to_txt(edu, '../.data/20250418.edu.txt')
to_txt(feat, '../.data/20250418.appfeat.txt')

