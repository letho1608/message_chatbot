import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from transformers import TextDataset, DataCollatorForLanguageModeling

def training(huggingface_model, output_model_path, messages_path, separator,
             block_size, epochs, train_batch_size, save_steps, save_total_limit):

    torch.cuda.is_available = lambda : False
    torch.set_grad_enabled(True)  

    tokenizer = AutoTokenizer.from_pretrained(huggingface_model)
    model = AutoModelForCausalLM.from_pretrained(huggingface_model)

    # Add the separator token to the tokenizer
    special_tokens_dict = {'additional_special_tokens': [separator]}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=messages_path,
        block_size=block_size
    )
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    training_args = TrainingArguments(
        output_dir='./results',
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=train_batch_size,
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        gradient_accumulation_steps=2,
        fp16=False,  # Disable mixed precision training
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
    )

    while True:
        try:
            trainer.train()
            break
        except torch.cuda.OutOfMemoryError as e:
            print(e)
            torch.cuda.empty_cache()
            input("Press Enter to retry")

    tokenizer.save_pretrained(output_model_path)
    model.save_pretrained(output_model_path)

if __name__ == "__main__":

    training(huggingface_model="sdadas/polish-gpt2-medium",
             output_model_path='./results/yourmodel',
             separator='(separator)',
             messages_path='fb_messages.txt',
             block_size=128,
             epochs=5,
             train_batch_size=4,
             save_steps=5000,
             save_total_limit=2)
