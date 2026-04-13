text="You sleep in the bed that you make not in the one that was made"
token=sorted(set(text.split()))
word2idx={}

#  create encoding set for each word and then reverse encoding
for i,w in enumerate(token):
    word2idx.update({w:i})
idx2wrd={}
for i,w in enumerate(token):
    idx2wrd.update({i:w})
print(word2idx)
# sentence encoding
encode=[]
for w in text.split():
    encode.append(word2idx[w])


# create tensor dataset for pytorch model
import torch
x,y=[],[]
for i in range(len(encode)-1):
    x.append(encode[i])
    y.append(encode[i+1])
x=torch.tensor(x)
y=torch.tensor(y)

print(token)
print(encode)