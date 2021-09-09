import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import string


def prediction(
        path,
        input,
        num_sentences=1,
        top_p=0.95,
        min_length=1,
        max_length=300
    ):
    model = GPT2LMHeadModel.from_pretrained(path)
    tokenizer = GPT2Tokenizer.from_pretrained(path)
    model.eval()
    outputs = []
    last_word = ''

    for _ in range(num_sentences):
        encoded_input = torch.tensor(tokenizer.encode(input)).unsqueeze(0)
        sample_outputs = model.generate(
            encoded_input,
            do_sample=True,
            min_length=min_length,
            max_length=max_length,
            top_p=top_p,
            num_return_sequences=1,
            top_k=30,
            temperature=0.9,
            repetition_penalty=2.0,
            no_repeat_ngram_size=2,
        )

        for sample_output in sample_outputs:
            sentence = tokenizer.decode(sample_output, skip_special_tokens=True)
        
        outputs.append(sentence.replace(last_word, "")) # replacing the prompt
        last_word = sentence.split(' ')[-1] # extract the last word to be used as prompt for the next sentence
        last_word = last_word.translate(str.maketrans('', '', string.punctuation))
        encoded_input = torch.tensor(tokenizer.encode(last_word)).unsqueeze(0)
        # print(f"{sentence=}\n{last_word}")
    
    result = " ".join(outputs).replace("1", "")
    return result


if __name__ == "__main__":
    path = "./training_checkpoints/gpt_lotr_09-09-0753/"
    prompt = "Gandalf was"
    result = prediction(path, prompt, num_sentences=5)
    print(result)
