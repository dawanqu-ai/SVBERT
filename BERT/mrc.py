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
import collections

from BERT.tokenization import BertTokenizer, BasicTokenizer
from BERT.modeling import BertForQuestionAnswering
from ZEN.modeling import ZenForQuestionAnswering
from ZEN.ngram_utils import ZenNgramDict

CONFIG_NAME = "config.json"
WEIGHTS_NAME = "pytorch_model.bin"
NGRAM_DICT_NAME = 'ngram.txt'

MODEL_NAME_DICT = {
    "ZEN": ZenForQuestionAnswering,
    "BERT": BertForQuestionAnswering
}

InputFeatures = collections.namedtuple(
    "InputFeatures",
    "unique_id text_index input_ids input_mask segment_ids ngram_ids ngram_positions "
    "token_to_orig_map tokens doc_tokens paragraph_text question_text token_is_max_context"
)
_DocSpan = collections.namedtuple(
    "DocSpan",
    ["start", "length"]
)
RawResult = collections.namedtuple(
    "RawResult",
    ["unique_id", "start_logits", "end_logits"]
)
_PrelimPrediction = collections.namedtuple(
    "PrelimPrediction",
    ["feature_index", "start_index", "end_index", "start_logit", "end_logit"]
)
_NbestPrediction = collections.namedtuple(
    "NbestPrediction",
    ["text", "start_logit", "end_logit"]
)

def is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False

def _compute_softmax(scores):
    """Compute softmax probability over raw logits."""
    if not scores:
        return []

    max_score = None
    for score in scores:
        if max_score is None or score > max_score:
            max_score = score

    exp_scores = []
    total_sum = 0.0
    for score in scores:
        x = math.exp(score - max_score)
        exp_scores.append(x)
        total_sum += x

    probs = []
    for score in exp_scores:
        probs.append(score / total_sum)
    return probs

class MrcInference():
    """
    Machine Reading Comprehension model
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
            model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../models/mrc")
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
            do_lower_case=self.config.get("do_lower_case", True)
        )
        self.model_name = self.config.get("model_name", "BERT")
        if self.model_name == "ZEN":
            if os.path.exists(ngram_freq_path) is False:
                print("ngram dict file not found! ")
                return
            self.ngram_dict = ZenNgramDict(ngram_freq_path, self.tokenizer)
        self.model = MODEL_NAME_DICT[self.model_name].from_pretrained(
            model_path
        ).to(self.device)

    def _inference(self, texts):
        model = self.model
        max_seq_length = self.config.get("max_seq_length", 384)
        max_query_length = self.config.get("max_query_length", 64)
        doc_stride = self.config.get("doc_stride", 128)

        model.eval()

        features = self.convert_text_to_feature(texts, max_seq_length, max_query_length, doc_stride)
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long).to(self.device)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long).to(self.device)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long).to(self.device)
        all_ngram_ids = torch.tensor([f.ngram_ids for f in features], dtype=torch.long).to(self.device)
        all_ngram_positions = torch.tensor([f.ngram_positions for f in features], dtype=torch.long).to(self.device)

        all_results = []
        with torch.no_grad():
            batch_start_logits, batch_end_logits = model(all_input_ids,
                           attention_mask=all_input_mask,
                           token_type_ids=all_segment_ids,
                           input_ngram_ids=all_ngram_ids,
                           ngram_position_matrix=all_ngram_positions)
            for i in range(all_input_ids.size(0)):
                all_results.append(RawResult(unique_id=features[i].unique_id,
                                             start_logits=batch_start_logits[i].detach().cpu().tolist(),
                                             end_logits=batch_end_logits[i].detach().cpu().tolist()))

        text_index_to_features = collections.defaultdict(list)
        for feature in features:
            text_index_to_features[feature.text_index].append(feature)

        unique_id_to_result = {}
        for result in all_results:
            unique_id_to_result[result.unique_id] = result

        answers = []
        for text_index,text in enumerate(texts):
            text_features = text_index_to_features[text_index]
            text_results = [unique_id_to_result[feature.unique_id] for feature in text_features]
            text_answer = self.answer_parsing(text_features, text_results)
            answers.append({
                "text": text,
                "answer": text_answer
            })
        return answers

    def answer_parsing(self, features, results):
        do_lower_case = self.config.get("do_lower_case", True)
        n_best_size = self.config.get("n_best_size", 20)
        max_answer_length = self.config.get("max_answer_length", 30)

        prelim_predictions = []
        for (feature_index, feature) in enumerate(features):
            result = results[feature_index]
            start_indexes = self._get_best_indexes(result.start_logits, n_best_size)
            end_indexes = self._get_best_indexes(result.end_logits, n_best_size)

            for start_index in start_indexes:
                for end_index in end_indexes:
                    if start_index >= len(feature.tokens):
                        continue
                    if end_index >= len(feature.tokens):
                        continue
                    if start_index not in feature.token_to_orig_map:
                        continue
                    if end_index not in feature.token_to_orig_map:
                        continue
                    if not feature.token_is_max_context.get(start_index, False):
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > max_answer_length:
                        continue
                    prelim_predictions.append(
                        _PrelimPrediction(
                            feature_index=feature_index,
                            start_index=start_index,
                            end_index=end_index,
                            start_logit=result.start_logits[start_index],
                            end_logit=result.end_logits[end_index]))
        prelim_predictions = sorted(
            prelim_predictions,
            key=lambda x: (x.start_logit + x.end_logit),
            reverse=True)

        seen_predictions = {}
        nbest = []
        for pred in prelim_predictions:
            if len(nbest) >= n_best_size:
                break
            feature = features[pred.feature_index]
            if pred.start_index > 0:  # this is a non-null prediction
                tok_tokens = feature.tokens[pred.start_index:(pred.end_index + 1)]
                orig_doc_start = feature.token_to_orig_map[pred.start_index]
                orig_doc_end = feature.token_to_orig_map[pred.end_index]
                orig_tokens = feature.doc_tokens[orig_doc_start:(orig_doc_end + 1)]
                tok_text = " ".join(tok_tokens)

                # De-tokenize WordPieces that have been split off.
                tok_text = tok_text.replace(" ##", "")
                tok_text = tok_text.replace("##", "")

                # Clean whitespace
                tok_text = tok_text.strip()
                tok_text = " ".join(tok_text.split())
                orig_text = " ".join(orig_tokens)

                final_text = self.get_final_text(tok_text, orig_text, do_lower_case)
                if final_text in seen_predictions:
                    continue

                seen_predictions[final_text] = True
            else:
                final_text = ""
                seen_predictions[final_text] = True

            nbest.append(
                _NbestPrediction(
                    text=final_text,
                    start_logit=pred.start_logit,
                    end_logit=pred.end_logit))

        if not nbest:
            return "empty"
        else:
            return nbest[0].text

    def get_final_text(self, pred_text, orig_text, do_lower_case):
        """Project the tokenized prediction back to the original text."""
        def _strip_spaces(text):
            ns_chars = []
            ns_to_s_map = collections.OrderedDict()
            for (i, c) in enumerate(text):
                if c == " ":
                    continue
                ns_to_s_map[len(ns_chars)] = i
                ns_chars.append(c)
            ns_text = "".join(ns_chars)
            return (ns_text, ns_to_s_map)

        tokenizer = BasicTokenizer(do_lower_case=do_lower_case)

        tok_text = " ".join(tokenizer.tokenize(orig_text))

        start_position = tok_text.find(pred_text)
        if start_position == -1:
            return orig_text
        end_position = start_position + len(pred_text) - 1

        (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
        (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

        if len(orig_ns_text) != len(tok_ns_text):
            return orig_text

        tok_s_to_ns_map = {}
        for (i, tok_index) in tok_ns_to_s_map.items():
            tok_s_to_ns_map[tok_index] = i

        orig_start_position = None
        if start_position in tok_s_to_ns_map:
            ns_start_position = tok_s_to_ns_map[start_position]
            if ns_start_position in orig_ns_to_s_map:
                orig_start_position = orig_ns_to_s_map[ns_start_position]

        if orig_start_position is None:
            return orig_text

        orig_end_position = None
        if end_position in tok_s_to_ns_map:
            ns_end_position = tok_s_to_ns_map[end_position]
            if ns_end_position in orig_ns_to_s_map:
                orig_end_position = orig_ns_to_s_map[ns_end_position]

        if orig_end_position is None:
            return orig_text

        output_text = orig_text[orig_start_position:(orig_end_position + 1)]
        return output_text

    def _check_is_max_context(self, doc_spans, cur_span_index, position):
        """Check if this is the 'max context' doc span for the token."""
        best_score = None
        best_span_index = None
        for (span_index, doc_span) in enumerate(doc_spans):
            end = doc_span.start + doc_span.length - 1
            if position < doc_span.start:
                continue
            if position > end:
                continue
            num_left_context = position - doc_span.start
            num_right_context = end - position
            score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
            if best_score is None or score > best_score:
                best_score = score
                best_span_index = span_index

        return cur_span_index == best_span_index

    def _get_best_indexes(self, logits, n_best_size):
        """Get the n-best logits from a list."""
        index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

        best_indexes = []
        for i in range(len(index_and_score)):
            if i >= n_best_size:
                break
            best_indexes.append(index_and_score[i][0])
        return best_indexes

    def _truncate_seq_pair(self, tokens_a, tokens_b, max_length):
        """Truncates a sequence pair in place to the maximum length."""
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()

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

    def convert_text_to_feature(self, texts, max_seq_length, max_query_length, doc_stride):
        tokenizer = self.tokenizer

        " code for bert char convert"
        features = []
        for text_index, text in enumerate(texts):
            paragraph_text, question_text = text

            doc_tokens = []
            char_to_word_offset = []
            prev_is_whitespace = True
            for c in paragraph_text:
                if is_whitespace(c):
                    prev_is_whitespace = True
                else:
                    if prev_is_whitespace:
                        doc_tokens.append(c)
                    else:
                        doc_tokens[-1] += c
                    prev_is_whitespace = False
                char_to_word_offset.append(len(doc_tokens) - 1)


            query_tokens = tokenizer.tokenize(question_text)
            if len(query_tokens) > max_query_length:
                query_tokens = query_tokens[0:max_query_length]

            tok_to_orig_index = []
            orig_to_tok_index = []
            all_doc_tokens = []
            for (i, token) in enumerate(doc_tokens):
                orig_to_tok_index.append(len(all_doc_tokens))
                sub_tokens = tokenizer.tokenize(token)
                for sub_token in sub_tokens:
                    tok_to_orig_index.append(i)
                    all_doc_tokens.append(sub_token)

            max_tokens_for_doc = max_seq_length - len(query_tokens) - 3


            doc_spans = []
            start_offset = 0
            while start_offset < len(all_doc_tokens):
                length = len(all_doc_tokens) - start_offset
                if length > max_tokens_for_doc:
                    length = max_tokens_for_doc
                doc_spans.append(_DocSpan(start=start_offset, length=length))
                if start_offset + length == len(all_doc_tokens):
                    break
                start_offset += min(length, doc_stride)

            for (doc_span_index, doc_span) in enumerate(doc_spans):
                tokens = []
                token_to_orig_map = {}
                token_is_max_context = {}
                segment_ids = []
                tokens.append("[CLS]")
                segment_ids.append(0)
                for token in query_tokens:
                    tokens.append(token)
                    segment_ids.append(0)
                tokens.append("[SEP]")
                segment_ids.append(0)

                for i in range(doc_span.length):
                    split_token_index = doc_span.start + i
                    token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]

                    is_max_context = self._check_is_max_context(doc_spans, doc_span_index, split_token_index)
                    token_is_max_context[len(tokens)] = is_max_context
                    tokens.append(all_doc_tokens[split_token_index])
                    segment_ids.append(1)
                tokens.append("[SEP]")
                segment_ids.append(1)

                input_ids = tokenizer.convert_tokens_to_ids(tokens)

                # The mask has 1 for real tokens and 0 for padding tokens. Only real
                # tokens are attended to.
                input_mask = [1] * len(input_ids)

                # Zero-pad up to the sequence length.
                while len(input_ids) < max_seq_length:
                    input_ids.append(0)
                    input_mask.append(0)
                    segment_ids.append(0)

                assert len(input_ids) == max_seq_length
                assert len(input_mask) == max_seq_length
                assert len(segment_ids) == max_seq_length

                if self.model_name == "ZEN":
                    ngram_ids, ngram_positions_matrix = self.extract_ngram(tokens, max_seq_length)
                else:
                    ngram_ids, ngram_positions_matrix = [], []

                features.append(
                    InputFeatures(
                        unique_id=len(features),
                        text_index=text_index,
                        input_ids=input_ids,
                        input_mask=input_mask,
                        segment_ids=segment_ids,
                        ngram_ids=ngram_ids,
                        ngram_positions=ngram_positions_matrix,
                        token_to_orig_map=token_to_orig_map,
                        tokens=tokens,
                        doc_tokens=doc_tokens,
                        paragraph_text=paragraph_text,
                        question_text=question_text,
                        token_is_max_context=token_is_max_context
                    )
                )
        return features

    def inference(self, context=None, question=None, texts=None, print_msg=True):
        if self.model is None:
            print("please load model first")
            return

        def check_texts_pairs():
            for text in texts:
                if len(text) != 2 or not isinstance(text[0], str) or not isinstance(text[1], str):
                    return False
            return True

        if isinstance(context, str) and isinstance(question, str):
            texts = [[context, question]]
            result = self._inference(texts)
            if print_msg is True:
                print(result)
            return result
        elif isinstance(texts, list) and check_texts_pairs():
            result = self._inference(texts)
            if print_msg is True:
                print(result)
            return result
        else:
            print("Input format error")
