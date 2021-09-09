import re
import json
from pathlib import Path


def regex_parse(input):
    text = re.sub('Mr\.', 'Mr ', input)                                          # Removes changes Mr. to Mr to avoid period confusion
    text = re.sub(r"(\.|\,|\;|\:|\!|\?)", lambda x: f' {x.group(1)} ', text)    # Adds space on both sides of punctuation
    text = re.sub(re.compile('[^\w,.!?:;\']+', re.UNICODE), ' ', text)          # Replaces all remaining non-alphanumeric/punctuation with space

    return text


def load_text():
    data_path = Path('data/')
    txt_files = list(data_path.rglob('*.txt'))
    lotr_full = ''
    for txt_file in txt_files:
        with open(txt_file, 'r') as file:
            text = regex_parse(file.read())
            lotr_full += text
            
    return lotr_full.split()

if __name__ == "__main__":
    with open('checkpoints/01-09-0508/misc/word_to_id.json') as json_file:
        word_to_id = json.load(json_file)

    should_capitalize = {word: True for word in word_to_id}

    lotr_full_text = load_text()

    for word in lotr_full_text:
        if word in should_capitalize:
            should_capitalize[word] = False



    should_capitalize['merry']     = True
    should_capitalize['balin']     = True
    should_capitalize['moria']     = True
    should_capitalize['i']         = True
    should_capitalize['bill']      = True
    should_capitalize['orthanc']   = True
    should_capitalize['butterbur'] = True


    with open('checkpoints/01-09-0344/misc/always_capitalized.json', 'w') as fp:
        json.dump(should_capitalize, fp, indent=4)
