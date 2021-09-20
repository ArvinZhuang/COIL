from typing import Optional

import torch
if torch.cuda.is_available():
    from torch.cuda.amp import autocast
from transformers import BertConfig, BertModel, BertTokenizer, PreTrainedModel
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

from arguments import ModelArguments, DataArguments, COILTrainingArguments as TrainingArguments
from marco_datasets import GroupedMarcoTrainDataset, MarcoPredDataset, MarcoEncodeDataset
from modeling import COIL
from transformers import AutoConfig, AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import (
    HfArgumentParser,
    set_seed,
)

from trainer import COILTrainer as Trainer



class QueryEncoder:
    def encode(self, text, **kwargs):
        pass


class UniCoilQueryEncoder(QueryEncoder):
    def __init__(self, model_name_or_path, tokenizer_name=None, device='cpu'):
        self.device = device
        self.model = UniCoilEncoder.from_pretrained(model_name_or_path, cache_dir='./cache')
        self.model.to(self.device)
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name or model_name_or_path)

    def encode(self, text, **kwargs):
        max_length = 128  # hardcode for now
        input_ids = self.tokenizer([text], max_length=max_length, padding='longest',
                                   truncation=True, add_special_tokens=True,
                                   return_tensors='pt').to(self.device)["input_ids"]
        batch_weights = self.model(input_ids).cpu().detach().numpy()
        batch_token_ids = input_ids.cpu().detach().numpy()
        return self._output_to_weight_dicts(batch_token_ids, batch_weights)[0]

    def _output_to_weight_dicts(self, batch_token_ids, batch_weights):
        to_return = []
        for i in range(len(batch_token_ids)):
            weights = batch_weights[i].flatten()
            tokens = self.tokenizer.convert_ids_to_tokens(batch_token_ids[i])
            # tokens = batch_token_ids[i]
            tok_weights = {}
            for j in range(len(tokens)):
                tok = str(tokens[j])
                weight = float(weights[j])
                # weight = int(np.ceil(float(weights[j]) / 5 * (2 ** 8)))
                if tok == '[CLS]':
                    continue
                if tok == '[PAD]':
                    break
                #
                # if tok == '101':
                #     continue
                # if tok == '0':
                #     break

                if tok not in tok_weights:
                    tok_weights[tok] = weight
                else:
                    tok_weights[tok] += weight
            to_return.append(tok_weights)
        return to_return


class UniCoilEncoder(PreTrainedModel):
    config_class = BertConfig
    base_model_prefix = "bert"
    load_tf_weights = None

    def __init__(self, config: BertConfig):
        super().__init__(config)
        self.config = config
        self.bert = BertModel(config)
        self.tok_proj = torch.nn.Linear(config.hidden_size, 1)
        self.init_weights()

    # Copied from transformers.models.bert.modeling_bert.BertPreTrainedModel._init_weights
    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, torch.nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, torch.nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def init_weights(self):
        self.bert.init_weights()
        # model_dict = torch.load(os.path.join("ckpts_uniCOIL_TILDE200/checkpoint-5epoch", 'model.pt'), map_location="cpu")
        self.tok_proj.apply(self._init_weights)
        # self.load_state_dict(model_dict, strict=False)

    def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
    ):
        input_shape = input_ids.size()
        device = input_ids.device
        if attention_mask is None:
            attention_mask = (
                torch.ones(input_shape, device=device)
                if input_ids is None
                else (input_ids != self.bert.config.pad_token_id)
            )
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        tok_weights = self.tok_proj(sequence_output)
        tok_weights = torch.relu(tok_weights)
        return tok_weights


# tokenizer = BertTokenizer.from_pretrained("ckpts_uniCOIL_TILDE200/checkpoint-5epoch")
# text = "what is information"
# max_length = 128  # hardcode for now
# input_ids = tokenizer([text], max_length=max_length, padding='longest',
#                            truncation=True, add_special_tokens=True,
#                            return_tensors='pt')["input_ids"]
#
# parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
#
# if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
#     # If we pass only one argument to the script and it's the path to a json file,
#     # let's parse it to get our arguments.
#     model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
# else:
#     model_args, data_args, training_args = parser.parse_args_into_dataclasses()
#     model_args: ModelArguments
#     data_args: DataArguments
#     training_args: TrainingArguments
#
# if (
#         os.path.exists(training_args.output_dir)
#         and os.listdir(training_args.output_dir)
#         and training_args.do_train
#         and not training_args.overwrite_output_dir
# ):
#     raise ValueError(
#         f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
#     )
#
# # Setup logging
# logging.basicConfig(
#     format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
#     datefmt="%m/%d/%Y %H:%M:%S",
#     level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
# )
#
# # Set seed
# # set_seed(training_args.seed)
#
# num_labels = 1
#
# config = AutoConfig.from_pretrained(
#     model_args.config_name if model_args.config_name else model_args.model_name_or_path,
#     num_labels=num_labels,
#     cache_dir=model_args.cache_dir,
# )
# tokenizer = AutoTokenizer.from_pretrained(
#     model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
#     cache_dir=model_args.cache_dir,
#     use_fast=False,
# )
# model = COIL.from_pretrained(
#     model_args, data_args, training_args,
#     model_args.model_name_or_path,
#     from_tf=bool(".ckpt" in model_args.model_name_or_path),
#     config=config,
#     cache_dir=model_args.cache_dir,
# )
#
# print(model.test_encode(input_ids).cpu().detach().numpy())
#


# model = UniCoilEncoder.from_pretrained("unicoil-tilde200-msmarco-passage")
#
#
# batch_weights = model(input_ids).cpu().detach().numpy()
# model.save_pretrained("unicoil-tilde200-msmarco-passage")
# tokenizer.save_pretrained("unicoil-tilde200-msmarco-passage")

# model


encoder = UniCoilQueryEncoder('ielab/unicoil-tilde200-msmarco-passage')

with open("topics.msmarco-passage-v2.dev.tsv", 'r') as f, open('topics.msmarco-passage.v2-dev.unicoil.tilde.expansion.tsv', 'w') as wf:
    lines = f.readlines()
    for line in tqdm(lines):
        qid, text = line.split('\t')
        query = []
        psg_tok_weight_dict = encoder.encode(text)
        for tok in psg_tok_weight_dict:
            freq = psg_tok_weight_dict[tok]
            freq = int(np.ceil(float(freq) / 5 * (2 ** 8)))
            query.extend([str(tok)] * freq)
        query = " ".join(query)
        if len(query) > 0:
            wf.write(f'{qid}\t{query}\n')
        else:
            wf.write(f'{qid}\t[SEP]')
