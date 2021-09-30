import os, glob
import json
from transformers import BertTokenizer
from tqdm import tqdm
from argparse import ArgumentParser




if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--file_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    args = parser.parse_args()

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    # file_path = '../msmarco_passage_unicoil_encoded_TILDE_200'
    # output_path = '../msmarco_passage_unicoil_encoded_TILDE_200_decoded'
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', use_fast=True, cache_dir="../cache")

    files = glob.glob(os.path.join(args.file_path, '*'))
    for file in tqdm(files):
        with open(file, 'r') as f, open(f"{args.output_path}/{file.split('/')[-1]}", 'w') as wf:
            for line in f:
                data = json.loads(line)
                vector = {}
                for tok_id in data['vector'].keys():
                    vector[tokenizer.decode([int(tok_id)])] = data['vector'][tok_id]
                data['vector'] = vector
                json.dump(data, wf)
                wf.write('\n')

