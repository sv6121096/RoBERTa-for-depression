from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch
import pandas as pd

tokenizer = RobertaTokenizer.from_pretrained("C:/Users/twitc/Downloads/project/results13") #loading the pre-saved model
model = RobertaForSequenceClassification.from_pretrained("C:/Users/twitc/Downloads/project/results12.1")

def predict(u): #the function that processes a given response and returns 0 or 1
    
    inputs = tokenizer(u, truncation=True, padding="max_length", max_length=256, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        prob = torch.sigmoid(logits) 
        predicted = 1 if prob >= 0.5 else 0
    return "0" if predicted == 0 else "1"

l = [] #a list of 0s and 1s where each statement is processed and a 0 or 1 is assigned to this list based on that
q= pd.read_csv("q3.csv") #loading a file of questions

def check(u): #some close ended questions can be answered in yes/no 
   if u=="no":
    l.append("0")
    print("0")
   elif u=="yes":
    l.append("1")
    print("1")
   else:
      l.append(predict(u))
      print(predict(u))

def check2(u):
   if u=="no":
    l.append("1")
    print("1")
   elif u=="yes":
    l.append("0")
    print("0")
   else:
      l.append(predict(u))
      print(predict(u))


s= input(f"Name\n") #Demographics
a= input(f"Age\n")
g= input(f"Gender\n")

for index, row in q.iterrows(): #questions will be asked one by one

    i = row['q']
    j = row['number']
    
    u = input(f'\n{j}.{i}\n').lower()
    
    if j in [3,11]: #close ended questions that are processed for a yes/no
      check2(u)
    elif j in [4, 6, 9, 10, 12, 13]:
      check(u)
    else:
      result = predict(u) 
#predict() calls the model and processes the inputs in case of open ended questions or elaborate answers
      l.append(result)
      print(result)


#final verdict according to the number of responses indicative of depression
if l.count("0")>l.count("1"):
   print("No indicators of depression")
   r= "N"
elif l.count("1")>l.count("0"):
   print("Indicators of depression are present")
   r= "Y"
elif l.count("1")==l.count("0"):
   print("Presence of depression indicators. However, further enquiry is suggested.")
   r= "MAYBE"
row = [[s, a, g, r]]
columns = ['Name', 'Age', 'Gender', "Indicators of depression"]

#a CSV file that saves data sorted in headings as given in columns array above
df = pd.DataFrame(row, columns=columns)
df.to_csv('record.csv',mode='a',header=False, index=False)
