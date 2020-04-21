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
import numpy as np
from collections import namedtuple

from .tokenization import BertTokenizer
from .modeling import BertForSequenceClassification

CONFIG_NAME = "config.json"
WEIGHTS_NAME = "pytorch_model.bin"

InputFeatures = namedtuple(
    "InputFeatures",
    "input_ids input_mask segment_ids")


class StsInference():
    """
    Sentence similarity computation inference model
    """
    def __init__(self, model_path=None, device="cpu"):
        self.model_path = model_path
        self.device = device
        self.model = None
        self.tokenizer = None
        self.config = None
        self.load_model(model_path)

    def load_model(self, model_path):
        if model_path is None:
            model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),"../models/sts")
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
            do_lower_case=self.config["do_lower_case"]
        )
        self.model = BertForSequenceClassification.from_pretrained(
            model_path,
            num_labels=self.config["num_labels"]
        ).to(self.device)

    def _inference(self, texts):
        model = self.model
        label_map = self.config["label_map"]
        max_seq_length = self.config["max_seq_length"]

        model.eval()

        features = self.convert_text_to_feature(texts, max_seq_length)
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long).to(self.device)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long).to(self.device)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long).to(self.device)

        with torch.no_grad():
            logits = model(all_input_ids, all_segment_ids, all_input_mask)

        def get_score(scores):
            scores = scores - scores.min()
            scores = scores / scores.sum()
            return dict([(label_map[str(i)], score) for i, score in enumerate(scores)])

        pred_scores = logits.detach().cpu().numpy()
        pred = np.argmax(pred_scores, axis=1)
        labels = [label_map[str(x)] for x in pred]
        pred_scores = [get_score(x) for x in pred_scores]

        return labels, pred_scores

    def _truncate_seq_pair(self, tokens_a, tokens_b, max_length):
        """Truncates a sequence pair in place to the maximum length."""

        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()

    def convert_text_to_feature(self, texts, max_seq_length):
        tokenizer = self.tokenizer

        " code for bert char convert"
        features = []
        for i, text in enumerate(texts):
            text_a, text_b = text
            tokens_a = tokenizer.tokenize(text_a)
            tokens_b = tokenizer.tokenize(text_b)
            self._truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
            tokens = ["[CLS]"] + tokens_a + ["[SEP]"] + tokens_b + ["[SEP]"]
            segment_ids = [0] * (len(tokens_a)+2) + [1] * (len(tokens_b) + 1)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding = [0] * (max_seq_length - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            features.append(
                InputFeatures(
                    input_ids=input_ids,
                    input_mask=input_mask,
                    segment_ids=segment_ids
                )
            )
        return features

    def inference(self, text_a=None, text_b=None, texts=None, print_msg=True):
        if self.model is None:
            print("please load model first")
            return

        def check_texts_pairs():
            for text in texts:
                if len(text) != 2 or not isinstance(text[0], str) or not isinstance(text[1], str):
                    return False
            return True

        if isinstance(text_a, str) and isinstance(text_b, str):
            texts = [[text_a,text_b]]
            labels, pred_scores = self._inference(texts)
            result = {
                "text": texts,
                "label": "unknown",
                "score": 0
            }
            if len(labels) > 0:
                result["label"] = labels[0]
                result["score"] = pred_scores[0]
            if print_msg is True:
                print(result)
            return result
        elif isinstance(texts, list) and check_texts_pairs():
            labels, pred_scores = self._inference(texts)
            result = [{
                "text": text,
                "label": label,
                "score": pred_score
            } for text, label, pred_score in zip(texts, labels, pred_scores)]
            if print_msg is True:
                print(result)
            return result
        else:
            print("Input format error")
