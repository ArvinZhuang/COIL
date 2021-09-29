from argparse import ArgumentParser
import os
import json
from tqdm import tqdm
import torch
from nltk.corpus import stopwords
import re
import numpy as np
from transformers import BertLMHeadModel, BertTokenizer
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def clean_vacab(tokenizer, do_stopwords=True):
    if do_stopwords:
        stop_words = set(stopwords.words('english'))
        # keep some common words in ms marco questions
        # stop_words.difference_update(["where", "how", "what", "when", "which", "why", "who"])
        stop_words.add("definition")

    vocab = tokenizer.get_vocab()
    tokens = vocab.keys()

    # bad_token = []
    bad_ids = []

    for stop_word in stop_words:
        ids = tokenizer(stop_word, add_special_tokens=False)["input_ids"]
        if len(ids) == 1:
            # bad_token.append(stop_word)
            bad_ids.append(ids[0])

    for token in tokens:
        token_id = vocab[token]
        if token_id in bad_ids:
            continue

        if token[0] == '#' and len(token) > 1:
            bad_ids.append(token_id)

        else:
            if not re.match("^[A-Za-z]*$", token):
                # bad_token.append(token)
                bad_ids.append(token_id)

    bad_ids.append(2015)  # add ##s to stopwords
    bad_ids.append(1011)  # add '-' to stopwords
    return bad_ids


def tilde_query_expansion(query, model, tokenizer, k, bad_ids):
    inputs = tokenizer.encode_plus(query, return_tensors='pt')
    inputs.input_ids[:, 0] = 0

    with torch.no_grad():
        outputs = model(**inputs.to(DEVICE), return_dict=True).logits[:, 0]
        probs = torch.sigmoid(outputs)
        selected = torch.topk(probs, k).indices.cpu().numpy()[0]

    expand_term_ids = np.setdiff1d(np.setdiff1d(selected, inputs.input_ids[0].cpu().numpy(), assume_unique=True),
                                   bad_ids, assume_unique=True)
    query.append(102)
    query.extend(expand_term_ids.tolist())
    # term = []
    # for id in expand_term_ids:
    #     term.append(tokenizer.decode([id]))
    # print(term)
    # print(tokenizer.decode(query))
    # print()



def main(args):
    model = BertLMHeadModel.from_pretrained("ielab/TILDE", cache_dir='./cache').eval().to(DEVICE)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', use_fast=True, cache_dir='./cache')
    bad_ids = clean_vacab(tokenizer)

    train_files = os.listdir(args.psg_train_dir)
    for file in tqdm(train_files, desc="writing files"):
        id_fout = open(f'{args.output_dir}/{file}', 'a+')
        with open(args.psg_train_dir + f"/{file}", 'r') as f:
            lines = f.readlines()
            for line in tqdm(lines):
                data = json.loads(line)
                pos_passages = []
                for pos_pass in data["pos"]:
                    pos_passages.append({'pid': pos_pass['pid'],
                                         'passage': pos_pass['passage']})
                neg_passages = []
                for neg_pass in data["neg"]:
                    neg_passages.append({'pid': neg_pass['pid'],
                                         'passage': neg_pass['passage']})

                tilde_query_expansion(data['qry']['query'], model, tokenizer, args.qry_k, bad_ids)

                temp = {
                    "qry": data['qry'],
                    "pos": pos_passages,
                    "neg": neg_passages
                }
                id_fout.write(f'{json.dumps(temp)}\n')

        id_fout.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--psg_train_dir', required=True)
    parser.add_argument('--qry_k', type=int, required=True)
    parser.add_argument('--output_dir', default='qry-psg-train-TILDE')
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    main(args)