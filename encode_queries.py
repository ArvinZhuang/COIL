# coding=utf-8
# Copyright 2021 COIL authors
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import sys
import pickle
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Union, Tuple
from tqdm import tqdm

import numpy as np
import torch
from timeit import default_timer as timer
from arguments import ModelArguments, DataArguments, COILTrainingArguments as TrainingArguments
from marco_datasets import GroupedMarcoTrainDataset, MarcoPredDataset, MarcoEncodeDataset
from modeling import COIL
from transformers import AutoConfig, AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import (
    HfArgumentParser,
    set_seed,
)


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))

    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args: ModelArguments
    data_args: DataArguments
    training_args: TrainingArguments


    # Set seed
    set_seed(training_args.seed)

    num_labels = 1

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        cache_dir=model_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=False,
    )
    model = COIL.from_pretrained(
        model_args, data_args, training_args,
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
    )


    model = model.to(training_args.device)
    model.eval()
    queires = []
    with open("queries.dev.small.sub.tsv", 'r') as f:
        for line in f:
            qid, query = line.split("\t")
            queires.append((qid, query))
    # lines = []
    total_time = 0
    for qid, query in tqdm(queires):
        tokenizer_start = timer()
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                inputs = tokenizer(query, return_tensors='pt')
                tok_ids = inputs['input_ids'][0][1:-1]
                for k, v in inputs.items():
                    inputs[k] = v.to(training_args.device)
                _, reps = model.encode(**inputs)
                reps = reps.cpu()[:, :, 0][0][1:-1]
                proccesed_query = []
                for i, rep in enumerate(reps):
                    cur_tok_weight = rep.item()
                    cur_tok_weight = 0 if cur_tok_weight < 0 else float(cur_tok_weight)
                    cur_tok_weight = int(np.ceil(cur_tok_weight / 5 * (2 ** 8)))
                    tok_id = tok_ids[i].item()

                    proccesed_query.extend([str(tok_id)] * cur_tok_weight)
                proccesed_query = " ".join(proccesed_query)
        tokenizer_end = timer()
        total_time += tokenizer_end - tokenizer_start
    print("Avg query processing time:%.1f ms" % ((1000 * total_time) / len(queires)))
                # print(proccesed_query)
                # lines.append(f'{qid}\t{proccesed_query}\n')

    # with open("queries.dev.small.sub.unicoil.tsv", 'w') as f:
    #     f.writelines(lines)


if __name__ == "__main__":
# python encode_queries.py \
# --model_name_or_path ckpts_uniCOIL_d2q/checkpoint-5epoch \
# --tokenizer_name bert-base-uncased \
# --output_dir test \
# --cache_dir cache \
# --token_dim 1 \
# --no_cls \
# --do_encode \
# --p_max_len 64 \
# --no_sep \
# --fp16 \
# --pooling max \
# --per_device_eval_batch_size 128 \
# --dataloader_num_workers 12

    main()
