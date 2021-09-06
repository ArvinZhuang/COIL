import json
from transformers import BertTokenizerFast

d2q_passages = []
with open("corpus_d2q/split00", 'rt') as fin:
    lines = fin.readlines()
    for line in lines[:50]:
        jcontent = json.loads(line)
        d2q_passages.append(jcontent['psg'])

TILDE_passages = []
with open("corpus_TILDE/split00", 'rt') as fin:
    lines = fin.readlines()
    for line in lines[:50]:
        jcontent = json.loads(line)
        TILDE_passages.append(jcontent['psg'])

tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased", cache_dir='./cache')
for i in range(50):
    print("------------------------")
    print("d2q: ", tokenizer.decode(d2q_passages[i]))
    print("TILDE: ", tokenizer.decode(TILDE_passages[i]))
    print("------------------------")