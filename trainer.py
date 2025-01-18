from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, TextDataset, DataCollatorForLanguageModeling


def load_dataset(file_path, tokenizer, block_size=128):
    dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=block_size
    )
    return dataset


def create_data_collator(tokenizer):
    return DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )


model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

file_path = "cleaned_poems_formatted.txt"
dataset = load_dataset(file_path, tokenizer)
data_collator = create_data_collator(tokenizer)

# Training Parameters
training_args = TrainingArguments(
    output_dir="gpt2-poetry-cleaned",
    overwrite_output_dir=True,
    num_train_epochs=5,
    per_device_train_batch_size=8,
    save_steps=500,
    save_total_limit=5,
    logging_dir="./logs",
    learning_rate=2e-5,
    fp16=True
)

# Training
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

trainer.train()

trainer.save_model("./gpt2-poetry-cleaned")
tokenizer.save_pretrained("./gpt2-poetry-cleaned")
