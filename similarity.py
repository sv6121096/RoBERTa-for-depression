import os
import torch
import pandas as pd
import numpy as np
import contractions
from transformers import RobertaTokenizerFast, RobertaModel
from sklearn.metrics.pairwise import cosine_similarity

model_path = "C:/Users/twitc/Downloads/project/results13"#load model
tokenizer = RobertaTokenizerFast.from_pretrained(model_path)
model = RobertaModel.from_pretrained(model_path)
model.eval()

# Load data
df = pd.read_csv("label.csv")

# Clean text
def clean_text(df):
    df["value"] = df["value"].str.replace(r'[,.()\[\]]', '', regex=True)
    df["value"] = df["value"].apply(contractions.fix)
    df["value"] = df["value"].str.lower()
    return df

df = clean_text(df)

text_0 = df[df["label"] == 0]["value"].tolist() #class separation
text_1 = df[df["label"] == 1]["value"].tolist()

def get_sentence_embeddings(sentences, batch_size=16, max_len=128):#sentence embeddings
    embeddings = []
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True,
                           max_length=max_len)
        with torch.no_grad():
            outputs = model(**inputs).last_hidden_state
        mask = inputs["attention_mask"].unsqueeze(-1).expand(outputs.size()).float()
        summed = torch.sum(outputs * mask, dim=1)
        count = torch.clamp(mask.sum(dim=1), min=1e-9)
        mean_pooled = (summed / count).cpu().numpy()
        embeddings.append(mean_pooled)
    return np.vstack(embeddings) if embeddings else np.array([])

emb_0 = get_sentence_embeddings(text_0)
emb_1 = get_sentence_embeddings(text_1)

def mean_intra_similarity(embeddings, label_name):#average intra-label cosine similarity
    if len(embeddings) < 2:
        print(f"[{label_name}] Not enough data.")
        return np.nan, None
    sim_matrix = cosine_similarity(embeddings)
    upper = np.triu_indices_from(sim_matrix, k=1)
    mean_sim = np.mean(sim_matrix[upper])
    print(f"\n[{label_name}] Cosine Similarity Matrix:\n{sim_matrix}")
    return mean_sim, sim_matrix

def mean_inter_similarity(emb0, emb1): #average inter-label cosine similarity
    if len(emb0) == 0 or len(emb1) == 0:
        print("[Inter] One of the embeddings is empty.")
        return np.nan, None
    sim_matrix = cosine_similarity(emb0, emb1)
    print(f"\n[Between Labels] Cosine Similarity Matrix:\n{sim_matrix}")
    return np.mean(sim_matrix), sim_matrix

sim_0, mat_0 = mean_intra_similarity(emb_0, "Label 0")
sim_1, mat_1 = mean_intra_similarity(emb_1, "Label 1")
sim_between, mat_between = mean_inter_similarity(emb_0, emb_1)

print(f"\n Within Label 0: {sim_0:.4f}")
print(f"Within Label 1: {sim_1:.4f}")
print(f"Between Labels: {sim_between:.4f}")
