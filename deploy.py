from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch
import pandas as pd

tokenizer = RobertaTokenizer.from_pretrained(
    "C:/Users/aryan/Downloads/RoBERTa-for-depression-main/checkpoint-246"
)

model = RobertaForSequenceClassification.from_pretrained(
    "C:/Users/aryan/Downloads/RoBERTa-for-depression-main/checkpoint-246"
)


def predict(u):

    inputs = tokenizer(
        u,
        truncation=True,
        padding="max_length",
        max_length=256,
        return_tensors="pt"
    )

    with torch.no_grad():

        outputs = model(**inputs)

        logits = outputs.logits

        predicted = torch.argmax(
            logits,
            dim=1
        ).item()

    return str(predicted)


l=[]

domain_scores = {
    "affect": [],
    "cognitive": [],
    "somatic": [],
    "social": [],
    "history": []
}


question_domain = {

    1:"affect",
    2:"social",
    3:"social",
    4:"affect",
    5:"somatic",
    6:"affect",
    7:"affect",
    8:"somatic",
    9:"somatic",
    10:"cognitive",
    11:"cognitive",
    12:"somatic",
    13:"history",
    14:"history",
    15:"history",
    16:"history",
    17:"history",
    18:"history"
}


q=pd.read_csv("q3.csv")


def check(u):

    if u=="no":
        print("0")
        return "0"

    elif u=="yes":
        print("1")
        return "1"

    else:
        p=predict(u)
        print(p)
        return p


def check2(u):

    if u=="no":
        print("1")
        return "1"

    elif u=="yes":
        print("0")
        return "0"

    else:
        p=predict(u)
        print(p)
        return p


s=input("Name\n")
a=input("Age\n")
g=input("Gender\n")


history_followup=False


for index,row in q.iterrows():

    i=row['q']
    j=row['number']


    # skip Q14-Q18 unless Q13 flagged positive
    if j in [14,15,16,17,18] and not history_followup:
        continue


    u=input(f"\n{j}.{i}\n").lower()


    if j in [3,11]:
        result=check2(u)

    elif j in [4,6,9,10,12,13,14,15,16,17,18]:
        result=check(u)

    else:
        result=predict(u)
        print(result)


    # determine whether history followup needed
    if j==13 and int(result)==1:
        history_followup=True


    l.append(result)

    domain=question_domain[j]
    domain_scores[domain].append(int(result))


print("\nDomain flags:\n")


for domain,vals in domain_scores.items():

    score=sum(vals)
    total=len(vals)

    print(f"{domain} -- {score}/{total}")


affect_ratio=sum(domain_scores["affect"])/len(domain_scores["affect"])

cognitive_ratio=sum(domain_scores["cognitive"])/len(domain_scores["cognitive"])

somatic_ratio=sum(domain_scores["somatic"])/len(domain_scores["somatic"])

social_ratio=sum(domain_scores["social"])/len(domain_scores["social"])

history_ratio= (
        sum(domain_scores["history"])/
        len(domain_scores["history"]))



if affect_ratio>=0.5 and cognitive_ratio>=0.5:

    print("\nIndicators of depression are present")
    r="Y"


elif (
(somatic_ratio>=0.5) and (social_ratio>=0.5 or
    history_ratio>=0.5)
) and (
    affect_ratio<0.5 and
    cognitive_ratio<0.5
):

    print(
        "\nFurther inquiry required into somatic complaints and patient history."
    )

    r="N"


elif (
    history_ratio>=0.5
) and (
    somatic_ratio<0.5 and
    social_ratio<0.5 and
    affect_ratio<0.5 and
    cognitive_ratio<0.5
):

    print(
        "\nNo indicators of depression at present but further inquiry into history of symptoms is suggested."
    )

    r="N"


else:

    print("\nNo indicators of depression")
    r="N"


row=[[s,a,g,r]]

columns=[
    'Name',
    'Age',
    'Gender',
    'Indicators of depression'
]

df=pd.DataFrame(row,columns=columns)

df.to_csv(
    'record.csv',
    mode='a',
    header=False,
    index=False
)
