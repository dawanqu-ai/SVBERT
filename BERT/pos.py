# coding: utf-8
# Copyright 2020 Sinovation Ventures AI Institute
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

from __future__ import absolute_import, division, print_function

import os
import torch
import json
import math
import numpy as np
from collections import namedtuple
import torch.nn.functional as F

from BERT.tokenization import BertTokenizer, BasicTokenizer, _is_punctuation
from BERT.modeling import BertForTokenClassification
from BERT.modeling_zen import ZenForTokenClassification
from BERT.ngram_utils import ZenNgramDict


CONFIG_NAME = "config.json"
WEIGHTS_NAME = "pytorch_model.bin"
NGRAM_DICT_NAME = 'ngram.txt'

MODEL_NAME_DICT = {
    "ZEN": ZenForTokenClassification,
    "BERT": BertForTokenClassification
}

InputFeatures = namedtuple(
    "InputFeatures",
    "input_ids input_mask segment_ids original_token ngram_ids ngram_positions valid_ids b_use_valid_filter")

def char_is_chinese_or_puntuation(char):
    if BasicTokenizer()._is_chinese_char(ord(char)) is True:
        return True
    if _is_punctuation(char) is True:
        return True
    return False

def is_alpha(ch):
    cp = ord(ch)
    if cp >= 65 and cp <= 90:
        return True
    if cp >= 97 and cp <= 122:
        return True
    return False

def pos_parse(text, valid, label):
    new_text = []
    token = ""
    for i in range(len(text)):
        if valid[i] == 0:
            token += text[i].replace("##","")
            continue
        if len(token) > 0:
            new_text.append(token)
        token = text[i].replace("##","")
    if len(token) > 0:
        new_text.append(token)
    text = new_text
    res = []
    for i in range(len(text)):
        res.append({
            "word": text[i],
            "label": label[i]
        })

    return res

class PosInference():
    """
    Part-of-speech tagging inference model
    """
    def __init__(self, model_path=None, device="cpu"):
        self.model_name = "BERT"
        self.model_path = model_path
        self.device = device
        self.model = None
        self.tokenizer = None
        self.config = None
        self.ngram_dict = None
        self.output_original_token = True
        self.load_model(model_path)

    def load_model(self, model_path):
        if model_path is None:
            model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),"../models/pos")
        config_file = os.path.join(model_path, CONFIG_NAME)
        model_bin = os.path.join(model_path, WEIGHTS_NAME)
        ngram_freq_path = os.path.join(model_path, NGRAM_DICT_NAME)
        if os.path.exists(model_path) is False or os.path.exists(model_bin) is False:
            print("model not found! ")
            return
        if os.path.exists(config_file) is False:
            print("config file not found! ")
            return

        self.config = json.load(open(config_file, 'r', encoding='utf-8'))
        self.tokenizer = BertTokenizer.from_pretrained(
            model_path,
            do_lower_case=self.config["do_lower_case"],
            output_original_token=self.output_original_token
        )
        self.model_name = self.config.get("model_name", "BERT")
        if self.model_name == "ZEN":
            if os.path.exists(ngram_freq_path) is False:
                print("ngram dict file not found! ")
                return
            self.ngram_dict = ZenNgramDict(ngram_freq_path, BertTokenizer.from_pretrained(
                model_path,
                do_lower_case=self.config["do_lower_case"],
                output_original_token=False
            ))
        self.model = MODEL_NAME_DICT[self.model_name].from_pretrained(
            model_path,
            num_labels=self.config["num_labels"]
        ).to(self.device)


    def _inference(self, texts):
        model = self.model
        label_map = self.config["label_map"]
        max_seq_length = self.config["max_seq_length"]

        model.eval()

        features = self.convert_text_to_feature(texts, max_seq_length)
        all_original_token = [f.original_token for f in features]
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long).to(self.device)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long).to(self.device)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long).to(self.device)
        all_valid_ids = torch.tensor([f.valid_ids for f in features], dtype=torch.long).to(self.device)
        all_ngram_ids = torch.tensor([f.ngram_ids for f in features], dtype=torch.long).to(self.device)
        all_ngram_positions = torch.tensor([f.ngram_positions for f in features], dtype=torch.long).to(self.device)
        all_b_use_valid_filter = [f.b_use_valid_filter for f in features]

        with torch.no_grad():
            logits = model(all_input_ids,
                           attention_mask=all_input_mask,
                           token_type_ids=all_segment_ids,
                           valid_ids=all_valid_ids,
                           input_ngram_ids=all_ngram_ids,
                           ngram_position_matrix=all_ngram_positions,
                           b_use_valid_filter=all_b_use_valid_filter)

        preds = torch.argmax(F.log_softmax(logits, dim=2), dim=2).detach().cpu().numpy()
        labels = [[label_map.get(str(x), "O") for x in pred] for pred in preds]
        return [pos_parse(text, valid[1:], label[1:]) for text, valid, label in zip(all_original_token, all_valid_ids.detach().cpu().numpy(), labels)]

    def extract_ngram(self, tokens, max_seq_length):
        ngram_matches = []
        #  Filter the ngram segment from 2 to max_ngram_len to check whether there is a ngram
        max_gram_n = self.ngram_dict.max_ngram_len
        for p in range(2, max_gram_n):
            for q in range(0, len(tokens) - p + 1):
                character_segment = tokens[q:q + p]
                # j is the starting position of the ngram
                # i is the length of the current ngram
                character_segment = tuple(character_segment)
                if character_segment in self.ngram_dict.ngram_to_id_dict:
                    ngram_index = self.ngram_dict.ngram_to_id_dict[character_segment]
                    ngram_matches.append([ngram_index, q, p, character_segment])

        # shuffle(ngram_matches)
        ngram_matches = sorted(ngram_matches, key=lambda s: s[0])

        max_ngram_in_seq_proportion = math.ceil((len(tokens) / max_seq_length) * self.ngram_dict.max_ngram_in_seq)
        if len(ngram_matches) > max_ngram_in_seq_proportion:
            ngram_matches = ngram_matches[:max_ngram_in_seq_proportion]

        ngram_ids = [ngram[0] for ngram in ngram_matches]
        ngram_positions = [ngram[1] for ngram in ngram_matches]
        ngram_lengths = [ngram[2] for ngram in ngram_matches]

        # record the masked positions
        ngram_positions_matrix = np.zeros(shape=(max_seq_length, self.ngram_dict.max_ngram_in_seq), dtype=np.int32)
        for i in range(len(ngram_ids)):
            ngram_positions_matrix[ngram_positions[i]:ngram_positions[i] + ngram_lengths[i], i] = 1.0

        # Zero-pad up to the max ngram in seq length.
        padding = [0] * (self.ngram_dict.max_ngram_in_seq - len(ngram_ids))
        ngram_ids += padding

        return ngram_ids, ngram_positions_matrix

    def convert_text_to_feature(self, texts, max_seq_length):
        tokenizer = self.tokenizer

        features = []
        b_use_valid_filter = False
        for _i, text in enumerate(texts):
            tokens, ori_tokens = [], []
            valid = []
            for i, sub_text in enumerate(text.split(" ")):
                sub_tokens, sub_ori_tokens = tokenizer.tokenize(sub_text)
                tokens.extend(sub_tokens)
                ori_tokens.extend(sub_ori_tokens)
                for m in range(len(sub_tokens)):
                    if m == 0:
                        valid.append(1)
                    else:
                        valid.append(0)
                        b_use_valid_filter = True
            if len(tokens) >= max_seq_length - 1:
                tokens = tokens[0:(max_seq_length - 2)]
                valid = valid[0:(max_seq_length - 2)]

            ntokens = []
            segment_ids = []
            ntokens.append("[CLS]")
            segment_ids.append(0)
            valid.insert(0, 1)
            for i, token in enumerate(tokens):
                ntokens.append(token)
                segment_ids.append(0)
            ntokens.append("[SEP]")
            segment_ids.append(0)
            valid.append(1)
            input_ids = tokenizer.convert_tokens_to_ids(ntokens)
            input_mask = [1] * len(input_ids)
            while len(input_ids) < max_seq_length:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)
                valid.append(1)

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            if self.model_name == "ZEN":
                ngram_ids, ngram_positions_matrix = self.extract_ngram(tokens, max_seq_length)
            else:
                ngram_ids, ngram_positions_matrix = [], []

            features.append(
                InputFeatures(
                    input_ids=input_ids,
                    input_mask=input_mask,
                    segment_ids=segment_ids,
                    original_token=ori_tokens,
                    ngram_ids=ngram_ids,
                    ngram_positions=ngram_positions_matrix,
                    valid_ids=valid,
                    b_use_valid_filter=b_use_valid_filter
                )
            )
        return features

    def inference(self, texts, print_msg=True):
        if self.model is None:
            print("please load model first")
            return

        if isinstance(texts, str):
            texts = [texts]
            labels = self._inference(texts)
            result = {
                "text":texts,
                "label":[]
            }
            if len(labels) > 0:
                result = labels[0]
            if print_msg is True:
                print(result)
            return result
        elif isinstance(texts, list):
            labels = self._inference(texts)
            result = [{
                "text":text,
                "label":label
            } for text,label in zip(texts, labels)]
            if print_msg is True:
                print(result)
            return result
        else:
            print("Input format error")