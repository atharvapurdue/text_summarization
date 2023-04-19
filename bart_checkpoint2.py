from datasets import load_dataset
import transformers
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorForSeq2Seq
import evaluate
import numpy as np

tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")

def tokenize_data(batch):
    pass

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)

    return {k: round(v, 4) for k, v in result.items()}


def main():
    
    dataset = load_dataset("cnn_dailymail")

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    rouge = evaluate.load("rouge")

    training_args = TrainingArguments(do_train=True)

    trainer = Trainer(model=model,
                      training_args=training_args,
                      eval_dataset=tokenized["te"])

if __name__=='__main__':
    main()