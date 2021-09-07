import os, glob
import json
from transformers import BertTokenizerFast
from tqdm import tqdm

file_path = '../msmarco_passage_unicoil_encoded_TILDE_200'
output_path = '../msmarco_passage_unicoil_encoded_TILDE_200_decoded'
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

files = glob.glob(os.path.join(file_path, '*'))
for file in tqdm(files):
    with open(file, 'r') as f, open(f"{output_path}/{file.split('/')[-1]}", 'w') as wf:
        for line in f:
            data = json.loads(line)
            vector = {}
            for tok_id in data['vector'].keys():
                vector[tokenizer.decode(int(tok_id))] = data['vector'][tok_id]
            data['vector'] = vector
            json.dump(data, wf)
            wf.write('\n')
    # print(file.split('/')[-1])