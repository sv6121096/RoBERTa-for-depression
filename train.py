from transformers import RobertaTokenizerFast, RobertaForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback, RobertaConfig
from datasets import Dataset
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import contractions
import seaborn as sns
import matplotlib.pyplot as plt
import os
os.makedirs("./results13", exist_ok=True) #results and logs directory
os.makedirs("./logs12.2", exist_ok=True)

tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base") #load the model
c = RobertaConfig.from_pretrained("roberta-base", num_labels=2, hidden_dropout_prob=0.2)
model = RobertaForSequenceClassification.from_pretrained("roberta-base", config=c)

data = pd.read_csv("label.csv") #training set
edata = pd.read_csv("elabel.csv") #test set

def clean_text(df): #the csv file has two headers- label (0 or 1) and value (statements)
    df["value"] = df["value"].str.replace(r'[,.()\[\]]', '', regex=True)
    df["value"] = df["value"].apply(contractions.fix)
    df["value"] = df["value"].str.lower()
    return df

data = clean_text(data)
edata = clean_text(edata)

majority = data[data["label"] == 0]
minority = data[data["label"] == 1]
munder = majority.sample(n=len(minority), random_state=7) #undersampling majority class 1:1 ratio
data_balanced = pd.concat([munder, minority]).sample(frac=1, random_state=7).reset_index(drop=True)

train_data, val_data = train_test_split(data_balanced, train_size=0.8, random_state=7)

train_data = Dataset.from_pandas(train_data)
val_data = Dataset.from_pandas(val_data)
elabel_data = Dataset.from_pandas(edata)

def preprocess_function(d):
    return tokenizer(d["value"], truncation=True, padding="max_length", max_length=256)


train_data = train_data.map(preprocess_function, batched=True)
val_data = val_data.map(preprocess_function, batched=True)
elabel_data = elabel_data.map(preprocess_function, batched=True)

#hyperparameters
training_args = TrainingArguments(
    output_dir="./results13",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=5,
    learning_rate=3e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    weight_decay=0.03,
    logging_dir="./logs12.2",
    logging_steps=20,
    warmup_steps=100,
    lr_scheduler_type="linear",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    fp16=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=val_data,
    tokenizer=tokenizer,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=1)]
)
trainer.train()

pred = np.argmax(trainer.predict(elabel_data).predictions, axis=1)

report = classification_report(edata["label"], pred) #performance metrics 
with open("./results13/classification_report.txt", "w") as f:
    f.write(report)

cm = confusion_matrix(edata["label"], pred) #confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Not Depressed", "Depressed"],
            yticklabels=["Not Depressed", "Depressed"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig("./results13/confusion_matrix.png")
plt.close()

model.save_pretrained("./results13")
tokenizer.save_pretrained("./results13")

print("FINISHED.")
