import pandas as pd
from sklearn.model_selection import train_test_split
import re
from tqdm import tqdm

df = pd.read_csv('booksummaries.txt', sep='\t', names=['wiki_id', 'freebase_id', 'title', 'author', 'date', 'genres', 'summary'])

train_test_ratio = 0.9
train_valid_ratio = 7/9
df_full_train, df_test = train_test_split(df, train_size = train_test_ratio, random_state = 1)
df_train, df_valid = train_test_split(df_full_train, train_size = train_valid_ratio, random_state = 1)


def build_dataset(df, dest_path):
    with open(dest_path, 'w', encoding='utf-8') as f:
        data = ''
        summaries = df['summary'].tolist()
        for summary in tqdm(summaries):
            summary = str(summary).strip()
            summary = re.sub(r"\s", " ", summary)
            bos_token = '<BOS>'
            eos_token = '<EOS>'
            data += bos_token + ' ' + summary + ' ' + eos_token + '\n'
            
        f.write(data)


if __name__ == "__main__":
    build_dataset(df_train, 'train.txt')
    build_dataset(df_valid, 'valid.txt')
    build_dataset(df_test, 'test.txt')
