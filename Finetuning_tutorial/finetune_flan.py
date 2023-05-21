"""
Description: Finetuning samsum dataset using Flan-T5 (tutorial from Phil Schmidt)
Author     : Tadbhagya Kumar
Date       : 5/20/2023 
"""

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import concatenate_datasets

# Dataset
dataset_id = "samsum"
dataset = load_dataset(dataset_id)

# dataset is split into train and test
# Each set is a dict with id, dialogue and summary 
print(f"len train: {len(dataset['train'])}, len test : {len(dataset['test'])}")
print(f"Example : {dataset['train'][0]}")  


# selected model
model_id = "google/flan-t5-base"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)

# tokenize input : dialogue
tokenized_inputs = concatenate_datasets([dataset["train"], dataset["test"]]).map(lambda x: tokenizer(x["dialogue"], truncation=True), batched=True, remove_columns= ["dialogue","summary"]) 
max_src_length = max([len(x) for x in tokenized_inputs["input_ids"]])
print(f"max length : {max_src_length}")

# tokenize outputs : summary 
tokenized_targets = concatenate_datasets([dataset["train"], dataset["test"]]).map(lambda x: tokenizer(x["summary"], truncation = True), batched = True, remove_columns = ["dialogue", "summary"])
max_trgt_length = max([len(x) for x in tokenized_targets["input_ids"]])
print(f"max target len: {max_trgt_length}")


## Preprocess function

def preprocess_function(sample, padding = "max_length"):

    # add prefix to data
    inputs = ["summarize : " + text for text in sample["dialogue"]]
    
    # Tokenize inputs
    model_inputs = tokenizer(inputs, max_length=max_src_length, padding=padding, truncation=True)
    # Tokenize labels
    labels = tokenizer(text_target = sample["summary"], max_length= max_trgt_length, padding=padding, truncation=True)
 
    # replacing padding with -100
    if padding == "max_length":
        labels["input_ids"] = [
            [(l if l!=tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]]
        

    model_inputs["labels"] = labels["input_ids"]

    return model_inputs


tokenized_dataset = dataset.map(preprocess_function,batched=True, remove_columns = ["dialogue","summary","id"])
print(f"Keys of tokenized dataset: {list(tokenized_dataset['train'].features)}")

## Model Training
model_id = "google/flan-t5-base"

model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

## Computation metrics
import evaluate
import nltk
import numpy as np
from nltk.tokenize import sent_tokenize
nltk.download("punkt")


# Metric
metric = evaluate.load("rouge")

def postprocess_text(preds, labels):

    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(sent_tokenize(label)) for label in labels]
    
    return preds, labels

def compute_metrics(eval_preds):

    preds, labels = eval_preds

    if isinstance(preds,tuple):
        preds = preds[0]

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    # metric
    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    result = {k: round(v * 100, 4) for k, v in result.items()}
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)

    return result


from transformers import DataCollatorForSeq2Seq

# we want to ignore tokenizer pad token in the loss
label_pad_token_id = -100
# Data collator
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    label_pad_token_id=label_pad_token_id,
    pad_to_multiple_of=8
)


from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

# Hugging Face repository id
repository_id = f"{model_id.split('/')[1]}-{dataset_id}"

# Define training args
training_args = Seq2SeqTrainingArguments(
    output_dir=repository_id,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    fp16=False, # Overflows with fp16
    learning_rate=5e-5,
    num_train_epochs=2,
    # logging & evaluation strategies
    logging_dir=f"{repository_id}/logs",
    logging_strategy="steps",
    logging_steps=500,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
)

# Create Trainer instance
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    compute_metrics=compute_metrics,
)

# Start training
trainer.train()