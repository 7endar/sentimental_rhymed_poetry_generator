import random
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import pronouncing
import re

# Loading the Model and the Tokenizer (gpt2-poetry-cleaned)
model = GPT2LMHeadModel.from_pretrained("gpt2-poetry-cleaned")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2-poetry-cleaned")

tokenizer.add_special_tokens({'pad_token': '[PAD]'})
model.resize_token_embeddings(len(tokenizer))


def get_last_word(text):
    tokens = re.sub(r'[^\w\s]', '', text).strip().split()
    return tokens[-1].lower() if tokens else ""


def is_rhyme(word_a, word_b):
    rhymes_of_a = set(pronouncing.rhymes(word_a))
    return (word_b in rhymes_of_a) or (word_a == word_b)


def generate_poem(input_ids, attention_mask, tokens_per_line, min_tokens=4, seed= None, val_num_return_sequences = 100, val_temperature = 1.1,
                  val_top_k = 55, val_top_p = 0.9, val_repetition_penalty = 1.1, val_do_sample = True ):


    # print(num_candidates)
    if seed is None:
        torch.manual_seed(random.randint(1, 100))
    else:
        torch.manual_seed(seed)

    outputs = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=tokens_per_line,
        min_new_tokens=min_tokens,
        num_return_sequences=val_num_return_sequences,
        temperature=val_temperature,
        top_k=val_top_k,
        top_p=val_top_p,
        repetition_penalty=val_repetition_penalty,
        do_sample=val_do_sample,
        pad_token_id=tokenizer.eos_token_id
    )
    return outputs

# USER INPUT, Write the first line for the poetry generator.
initial_prompt = "all the places i want to see"

# How many lines?
num_lines = 4

# How many tokens for each line?
tokens_per_line = 10


current_text = initial_prompt.strip()
print(f"Line 1: {current_text}")


outputs = []
for line_idx in range(2, num_lines + 1):
    previous_line_last_word = get_last_word(current_text)

    valid_candidates = []
    counter = 0
    while (not valid_candidates) and counter < 10:
        counter += 1

        inputs = tokenizer(
            current_text + "\n",
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
            return_attention_mask=True
        )

        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        outputs = generate_poem(input_ids, attention_mask, tokens_per_line)

        for out_ids in outputs:
            full_text = tokenizer.decode(out_ids, skip_special_tokens=True)
            candidate_last_word = get_last_word(full_text)

            if is_rhyme(previous_line_last_word, candidate_last_word):
                # print(candidate_last_word)
                valid_candidates.append(full_text)
                # print(len(valid_candidates))

    # print(len(valid_candidates[0]))
    if not valid_candidates:
        # print("no rhyme!!!")
        valid_candidates.append(tokenizer.decode(outputs[0], skip_special_tokens=True))

    chosen_text = valid_candidates[0]  # Select the first valid candidate

    chosen_lines = [line.strip() for line in chosen_text.split("\n") if line.strip()]
    # print(f"chosen lines: {chosen_lines}")
    new_line = chosen_lines[-1].strip()
    # print(f"new line: {new_line}")

    current_text += "\n" + new_line

    print(f"Line {line_idx}: {new_line}")
