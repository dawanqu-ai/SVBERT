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
from collections import namedtuple
import torch.nn.functional as F

from .tokenization import BertTokenizer
from .modeling import BertForTokenClassification

CONFIG_NAME = "config.json"
WEIGHTS_NAME = "pytorch_model.bin"

InputFeatures = namedtuple(
    "InputFeatures",
    "input_ids input_mask segment_ids original_token")

def ner_parse(text,label):
    words_list = []
    word_idx = 0
    for i in range(len(text)):
        if text[i][:2] == "##" and len(words_list) > 0:
            words_list[-1]["subword"].append(text[i][2:])
        else:
            words_list.append({
                "subword":[text[i]],
                "label": label[i],
                "position": word_idx
            })
            word_idx += 1
    for word in words_list:
        word["word"] = ''.join(word["subword"])

    nes_list = []
    for i, word in enumerate(words_list):
        if word["label"][:1] not in ["B", "I"]:
            continue
        ne = {
            "type": word["label"][2:],
            "bi": word["label"][:1],
            "word": word["word"],
            "position": word["position"],
            "index": i
        }

        if ne["bi"] == "I" and len(nes_list) > 0 and \
                nes_list[-1]["index"] + 1 == i and nes_list[-1]["type"] == ne["type"]:
            nes_list[-1]["word"] = "{} {}".format(nes_list[-1]["word"], ne["word"])
            nes_list[-1]["index"] = i
        else:
            nes_list.append(ne)

    return [{
        "entity": ne["word"],
        "type": ne["type"],
        "position": ne["position"]
    } for ne in nes_list]

class NerInference():
    """
    Name entity recognition inference model
    """
    def __init__(self, model_path=None, device="cpu"):
        self.model_path = model_path
        self.device = device
        self.model = None
        self.tokenizer = None
        self.config = None
        self.output_original_token = True
        self.load_model(model_path)

    def load_model(self, model_path):
        if model_path is None:
            model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),"../models/ner")
        config_file = os.path.join(model_path, CONFIG_NAME)
        model_bin = os.path.join(model_path, WEIGHTS_NAME)
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
        self.model = BertForTokenClassification.from_pretrained(
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

        with torch.no_grad():
            logits = model(all_input_ids, all_segment_ids, all_input_mask)

        preds = torch.argmax(F.log_softmax(logits, dim=2), dim=2).detach().cpu().numpy()
        labels = [[label_map.get(str(x), "O") for x in pred] for pred in preds]
        return [ner_parse(text, label[1:]) for text, label in zip(all_original_token, labels)]

    def convert_text_to_feature(self, texts, max_seq_length):
        tokenizer = self.tokenizer

        " code for bert char convert"
        features = []
        for _i, text in enumerate(texts):
            tokens, ori_tokens = tokenizer.tokenize(text)
            if len(tokens) >= max_seq_length - 1:
                tokens = tokens[0:(max_seq_length - 2)]

            ntokens = []
            segment_ids = []
            ntokens.append("[CLS]")
            segment_ids.append(0)
            for i, token in enumerate(tokens):
                ntokens.append(token)
                segment_ids.append(0)
            ntokens.append("[SEP]")
            segment_ids.append(0)
            input_ids = tokenizer.convert_tokens_to_ids(ntokens)
            input_mask = [1] * len(input_ids)
            while len(input_ids) < max_seq_length:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            features.append(
                InputFeatures(
                    input_ids=input_ids,
                    input_mask=input_mask,
                    segment_ids=segment_ids,
                    original_token=ori_tokens
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
            result = "unknown"
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