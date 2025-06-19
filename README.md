# RoBERTa for predicting depression

This project involves training and deploying for interview-style dialogue analysis and classification in case of detecting depression.
The model has been trained on DAIC-WOZ dataset which is not available in public domain but upon reasonable request.
The current model has achieved a macro avg F1 of 0.82 
---

## Dataset Source

- **DAIC-WOZ Dataset:**  
  [https://dcapswoz.ict.usc.edu/](https://dcapswoz.ict.usc.edu/)  
---

## Learning Resources

These resources were used to better understand both theory and coding:

- [HuggingFace LLM Course](https://huggingface.co/learn/llm-course/chapter0/1?fw=pt) — Foundation on transformers and model development  
- [Captum: BERT SQuAD Interpretability Tutorial](https://captum.ai/tutorials/Bert_SQUAD_Interpret) — Token attribution and interpretability  
- [Semantic Textual Similarity with BERT](https://medium.com/@Mustafa77/semantic-textual-similarity-with-bert-e10355ed6afa) — Blog explanation of semantic similarity using BERT  
- [Fine-Tuning RoBERTa for Topic Classification](https://achimoraites.medium.com/fine-tuning-roberta-for-topic-classification-with-hugging-face-transformers-and-datasets-library-c6f8432d0820)  
- [Deep Learning with Python (PDF)](https://sourestdeeds.github.io/pdf/Deep%20Learning%20with%20Python.pdf) — Reference material for implementation  

---

## File Descriptions

| File / Directory     | Description |
|----------------------|-------------|
| `train.py`           | Script to train and validate the model. Saves performance metrics and confusion matrix in the working directory. |
| `attribution.py`     | Computes token-level attribution scores for given text inputs using interpretability methods. |
| `similarity.py`      | Calculates average semantic similarity scores both within and between the two defined classes. |
| `q3.txt`             | Contains a list of predefined questions used during the evaluation or interview process. |
| `deploy/`            | Module to load and run the fine-tuned model for real-time interview-style interaction (text only). |
