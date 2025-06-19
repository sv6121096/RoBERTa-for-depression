from transformers import RobertaTokenizer, RobertaForSequenceClassification
from captum.attr import LayerIntegratedGradients
import torch
import matplotlib.pyplot as plt

device = "cpu"
model_path = "C:/Users/twitc/Downloads/project/results12.1/checkpoint-246" # Load model and tokenizer
model = RobertaForSequenceClassification.from_pretrained(model_path).to(device)
tokenizer = RobertaTokenizer.from_pretrained(model_path)
model.eval()

text = ("ABC..")# Interview text 
inputs = tokenizer(text, return_tensors="pt", truncation=True)# tokenize input
input_ids = inputs["input_ids"].to(device)
attention_mask = inputs["attention_mask"].to(device)

ref_input_ids = torch.full_like(input_ids, tokenizer.pad_token_id)# create baseline

tokens = tokenizer.convert_ids_to_tokens(input_ids[0])# convert token ids to actual tokens for labeling later
tokens = [t.replace("Ġ", "") for t in tokens] 

logits = model(input_ids=input_ids, attention_mask=attention_mask).logits # get predicted class
probs = torch.softmax(logits, dim=1)
pred_label = torch.argmax(probs, dim=1).item()

def forward_func(inputs_embeds, attention_mask):
    return torch.softmax(model(inputs_embeds=inputs_embeds, attention_mask=attention_mask).logits, dim=1)

input_embed = model.roberta.embeddings(input_ids)# get embeddings for input and baseline
ref_embed = model.roberta.embeddings(ref_input_ids)

lig = LayerIntegratedGradients(forward_func, model.roberta.embeddings)
attributions, delta = lig.attribute(
    inputs=input_embed,
    baselines=ref_embed,
    additional_forward_args=(attention_mask,),
    target=pred_label,
    return_convergence_delta=True
)

attributions_sum = attributions.sum(dim=-1).squeeze(0)
attributions_sum = attributions_sum / torch.norm(attributions_sum)
scores = attributions_sum.detach().cpu().tolist() # match tokens and scores, sort by absolute attribution
top_tokens_scores = sorted(zip(tokens, scores), key=lambda x: abs(x[1]), reverse=True)[:10]
top_tokens, top_scores = zip(*top_tokens_scores)

plt.figure(figsize=(10, 5))# plot the results
plt.bar(
    range(len(top_tokens)),
    top_scores,
    tick_label=top_tokens,
    color=["skyblue" if s > 0 else "pink" for s in top_scores]
)
plt.xticks(rotation=45, ha='right')
plt.ylabel("Attribution Score")
plt.title(f"Predicted Label: {pred_label}")
plt.tight_layout()
plt.savefig("00ATT_top10(2).png")
plt.close()

print("FINISHED")
