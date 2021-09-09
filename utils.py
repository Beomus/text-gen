import datetime
import re
from pathlib import Path
import torch
from torch.utils.data import TensorDataset, Dataset
from collections import Counter


def format_time(elapsed):
    return str(datetime.timedelta(seconds=int(round((elapsed)))))


def parse_and_clean(input):
        text = input.lower()                                                        # Maps to lowercase
        text = re.sub('mr\.', 'mr ', text)                                          # Removes changes Mr. to Mr to avoid period confusion
        text = re.sub(r"(\.|\,|\;|\:|\!|\?)", lambda x: f' {x.group(1)} ', text)    # Adds space on both sides of punctuation
        text = re.sub(re.compile('[^\w,.!?:;\']+', re.UNICODE), ' ', text)          # Replaces all remaining non-alphanumeric/punctuation with space

        return text


class LstmData:
    def __init__(self, dataset, min_occurences, n_pred):
        self.min_occurences = min_occurences
        self.n_pred = n_pred
        self.full_text = self._load_full_text(dataset=dataset)
        self.word_to_id, self.id_to_word = self.get_vocab()
        self.full_ids = [self.word_to_id[word] for word in self.full_text]
        self.train_data = self.get_tensor_dataset()


    def _load_full_text(self, dataset='lotr'):
        data_path = Path(f'data/{dataset}')
        txt_files = list(data_path.rglob('*.txt'))
        full_text = ''
        for txt_file in txt_files:
            with open(txt_file, 'r', encoding="utf8") as f:
                f = f.read()
                text = parse_and_clean(f)
                full_text += text

        return full_text.split()


    def get_vocab(self):
        vocab = Counter()
        for word in self.full_text:
            vocab[word] += 1

        vocab_top = Counter({k: c for k, c in vocab.items() if c > self.min_occurences})
        vocab_tuples = vocab_top.most_common(len(vocab_top))

        word_to_id = Counter({word: i+1 for i,(word, c) in enumerate(vocab_tuples)})
        id_to_word = ["_"] + [word for word, index in word_to_id.items()]

        return word_to_id, id_to_word


    def get_tensor_dataset(self):
        features = []
        labels = []
        for i in range(self.n_pred, len(self.full_ids)):
            labels.append(self.full_ids[i])
            features.append(self.full_ids[i-self.n_pred:i])

        return TensorDataset(torch.tensor(features), torch.tensor(labels))


class GptData(Dataset):
    def __init__(self, dataset, tokenizer, max_length=None):
        self.tokenizer = tokenizer
        self.input_ids = []
        self.attn_masks = []

        self.sentence_list = self._get_sentence_list(dataset, max_length)
        self.max_token_length = self._get_max_token_length()

        for sentence in self.sentence_list:
            encodings_dict = tokenizer(
                '<|startoftext|>'+ sentence + '<|endoftext|>', 
                truncation=True, 
                max_length=self.max_token_length, 
                padding="max_length"
            )

            self.input_ids.append(torch.tensor(encodings_dict['input_ids']))
            self.attn_masks.append(torch.tensor(encodings_dict['attention_mask']))


    def _get_sentence_list(self, dataset, max_length):
        data_path = Path(f'data/{dataset}')
        txt_files = list(data_path.rglob('*.txt'))
        full_text = ''
        for txt_file in txt_files:
            with open(txt_file, 'r', encoding="utf8") as f:
                text = f.read()
                full_text += text
        sentence_list = re.split(r'(?<=\.) ', full_text)
        if max_length:
            sentence_list = [sentence for sentence in sentence_list if len(sentence) < max_length]
        return sentence_list

    def _get_max_token_length(self):
        return max([len(self.tokenizer.encode(sentence)) for sentence in self.sentence_list])
    

    def __len__(self):
        return len(self.input_ids)
    
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.attn_masks[idx]


if __name__ == "__main__":
    from transformers import GPT2Tokenizer

    tokenizer = GPT2Tokenizer.from_pretrained(
        'gpt2', 
        bos_token='<|startoftext|>', 
        eos_token='<|endoftext|>', 
        pad_token='<|pad|>'
    )

    dataset = GptData('hp', tokenizer)
    print(dataset.max_token_length)
    print(len(dataset))
    print(dataset[10])
