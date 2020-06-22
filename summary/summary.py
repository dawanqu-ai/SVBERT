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
import re
import logging
import torch
import json
import math
import cupy as np
import networkx as nx
import codecs
from itertools import product
from collections import Counter
from gensim.models.keyedvectors import KeyedVectors

cut_word_punct_pattern = r'([!\"\$%\'\(\)\*\+,\.:;\-<=·>?@\[\\\]\_ـ`{\|}~—٪’،؟`୍“؛”ۚ【»؛\s+«–…‘])'
def stop_words():
        filepath = "ar_stopwords.txt"
        stopwords = set([line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()])
        return stopwords

def tokenizer( content , stop_words_filter , stop_words):
    total_cutword = []
    words = re.split(cut_word_punct_pattern , content)
    words = [word for word in words if word not in set([""," "])]
    if stop_words_filter:
        words = list(filter(lambda x: x not in stop_words, words))
    return words
def cut_sentence(sentence):
    """
    :param sentence:str
    :return:list
    """
    para = re.sub('([。！？؟.\?\n])([^”’])', r"\1\n\2", sentence)
    para = re.sub('(\.{6}\n\r)([^”’])', r"\1\n\2", para)
    para = re.sub('(\…{2}\r)([^”’])', r"\1\n\2", para) 
    para = re.sub('([。！？؟.\?][”’])([^，。！؟？.\?])', r'\1\n\2', para)
    para = para.rstrip()
    return para.split("\n")
stop_words = set([
'،',
'ء',
'ءَ',
'آ',
'آب',
'آذار',
'آض',
'آل',
'آمينَ',
'آناء',
'آنفا',
'آه',
'آهاً',
'آهٍ',
'آهِ',
'أ',
'أبدا',
'أبريل',
'أبو',
'أبٌ',
'أجل',
'أجمع',
'أحد',
'أخبر',
'أخذ',
'أخو',
'أخٌ',
'أربع',
'أربعاء',
'أربعة',
'أربعمئة',
'أربعمائة',
'أرى',
'أسكن',
'أصبح',
'أصلا',
'أضحى',
'أطعم',
'أعطى',
'أعلم',
'أغسطس',
'أفريل',
'أفعل به',
'أفٍّ',
'أقبل',
'أكتوبر',
'أل',
'ألا',
'ألف',
'ألفى',
'أم',
'أما',
'أمام',
'أمامك',
'أمامكَ',
'أمد',
'أمس',
'أمسى',
'أمّا',
'أن',
'أنا',
'أنبأ',
'أنت',
'أنتم',
'أنتما',
'أنتن',
'أنتِ',
'أنشأ',
'أنه',
'أنًّ',
'أنّى',
'أهلا',
'أو',
'أوت',
'أوشك',
'أول',
'أولئك',
'أولاء',
'أولالك',
'أوّهْ',
'أى',
'أي',
'أيا',
'أيار',
'أيضا',
'أيلول',
'أين',
'أيّ',
'أيّان',
'أُفٍّ',
'ؤ',
'إحدى',
'إذ',
'إذا',
'إذاً',
'إذما',
'إذن',
'إزاء',
'إلى',
'إلي',
'إليكم',
'إليكما',
'إليكنّ',
'إليكَ',
'إلَيْكَ',
'إلّا',
'إمّا',
'إن',
'إنَّ',
'إى',
'إياك',
'إياكم',
'إياكما',
'إياكن',
'إيانا',
'إياه',
'إياها',
'إياهم',
'إياهما',
'إياهن',
'إياي',
'إيهٍ',
'ئ',
'ا',
'ا?',
'ا?ى',
'االا',
'االتى',
'ابتدأ',
'ابين',
'اتخذ',
'اثر',
'اثنا',
'اثنان',
'اثني',
'اثنين',
'اجل',
'احد',
'اخرى',
'اخلولق',
'اذا',
'اربعة',
'اربعون',
'اربعين',
'ارتدّ',
'استحال',
'اصبح',
'اضحى',
'اطار',
'اعادة',
'اعلنت',
'اف',
'اكثر',
'اكد',
'الآن',
'الألاء',
'الألى',
'الا',
'الاخيرة',
'الان',
'الاول',
'الاولى',
'التى',
'التي',
'الثاني',
'الثانية',
'الحالي',
'الذاتي',
'الذى',
'الذي',
'الذين',
'السابق',
'الف',
'اللاتي',
'اللتان',
'اللتيا',
'اللتين',
'اللذان',
'اللذين',
'اللواتي',
'الماضي',
'المقبل',
'الوقت',
'الى',
'الي',
'اليه',
'اليها',
'اليوم',
'اما',
'امام',
'امس',
'امسى',
'ان',
'انبرى',
'انقلب',
'انه',
'انها',
'او',
'اول',
'اي',
'ايار',
'ايام',
'ايضا',
'ب',
'بؤسا',
'بإن',
'بئس',
'باء',
'بات',
'باسم',
'بان',
'بخٍ',
'بد',
'بدلا',
'برس',
'بسبب',
'بسّ',
'بشكل',
'بضع',
'بطآن',
'بعد',
'بعدا',
'بعض',
'بغتة',
'بل',
'بلى',
'بن',
'به',
'بها',
'بهذا',
'بيد',
'بين',
'بَسْ',
'بَلْهَ',
'ة',
'ت',
'تاء',
'تارة',
'تاسع',
'تانِ',
'تانِك',
'تبدّل',
'تجاه',
'تحت',
'تحوّل',
'تخذ',
'ترك',
'تسع',
'تسعة',
'تسعمئة',
'تسعمائة',
'تسعون',
'تسعين',
'تشرين',
'تعسا',
'تعلَّم',
'تفعلان',
'تفعلون',
'تفعلين',
'تكون',
'تلقاء',
'تلك',
'تم',
'تموز',
'تينك',
'تَيْنِ',
'تِه',
'تِي',
'ث',
'ثاء',
'ثالث',
'ثامن',
'ثان',
'ثاني',
'ثلاث',
'ثلاثاء',
'ثلاثة',
'ثلاثمئة',
'ثلاثمائة',
'ثلاثون',
'ثلاثين',
'ثم',
'ثمان',
'ثمانمئة',
'ثمانون',
'ثماني',
'ثمانية',
'ثمانين',
'ثمنمئة',
'ثمَّ',
'ثمّ',
'ثمّة',
'ج',
'جانفي',
'جدا',
'جعل',
'جلل',
'جمعة',
'جميع',
'جنيه',
'جوان',
'جويلية',
'جير',
'جيم',
'ح',
'حاء',
'حادي',
'حار',
'حاشا',
'حاليا',
'حاي',
'حبذا',
'حبيب',
'حتى',
'حجا',
'حدَث',
'حرى',
'حزيران',
'حسب',
'حقا',
'حمدا',
'حمو',
'حمٌ',
'حوالى',
'حول',
'حيث',
'حيثما',
'حين',
'حيَّ',
'حَذارِ',
'خ',
'خاء',
'خاصة',
'خال',
'خامس',
'خبَّر',
'خلا',
'خلافا',
'خلال',
'خلف',
'خمس',
'خمسة',
'خمسمئة',
'خمسمائة',
'خمسون',
'خمسين',
'خميس',
'د',
'دال',
'درهم',
'درى',
'دواليك',
'دولار',
'دون',
'دونك',
'ديسمبر',
'دينار',
'ذ',
'ذا',
'ذات',
'ذاك',
'ذال',
'ذانك',
'ذانِ',
'ذلك',
'ذهب',
'ذو',
'ذيت',
'ذينك',
'ذَيْنِ',
'ذِه',
'ذِي',
'ر',
'رأى',
'راء',
'رابع',
'راح',
'رجع',
'رزق',
'رويدك',
'ريال',
'ريث',
'رُبَّ',
'ز',
'زاي',
'زعم',
'زود',
'زيارة',
'س',
'ساء',
'سابع',
'سادس',
'سبت',
'سبتمبر',
'سبحان',
'سبع',
'سبعة',
'سبعمئة',
'سبعمائة',
'سبعون',
'سبعين',
'ست',
'ستة',
'ستكون',
'ستمئة',
'ستمائة',
'ستون',
'ستين',
'سحقا',
'سرا',
'سرعان',
'سقى',
'سمعا',
'سنة',
'سنتيم',
'سنوات',
'سوف',
'سوى',
'سين',
'ش',
'شباط',
'شبه',
'شتانَ',
'شخصا',
'شرع',
'شمال',
'شيكل',
'شين',
'شَتَّانَ',
'ص',
'صاد',
'صار',
'صباح',
'صبر',
'صبرا',
'صدقا',
'صراحة',
'صفر',
'صهٍ',
'صهْ',
'ض',
'ضاد',
'ضحوة',
'ضد',
'ضمن',
'ط',
'طاء',
'طاق',
'طالما',
'طرا',
'طفق',
'طَق',
'ظ',
'ظاء',
'ظل',
'ظلّ',
'ظنَّ',
'ع',
'عاد',
'عاشر',
'عام',
'عاما',
'عامة',
'عجبا',
'عدا',
'عدة',
'عدد',
'عدم',
'عدَّ',
'عسى',
'عشر',
'عشرة',
'عشرون',
'عشرين',
'عل',
'علق',
'علم',
'على',
'علي',
'عليك',
'عليه',
'عليها',
'علًّ',
'عن',
'عند',
'عندما',
'عنه',
'عنها',
'عوض',
'عيانا',
'عين',
'عَدَسْ',
'غ',
'غادر',
'غالبا',
'غدا',
'غداة',
'غير',
'غين',
'ـ',
'ف',
'فإن',
'فاء',
'فان',
'فانه',
'فبراير',
'فرادى',
'فضلا',
'فقد',
'فقط',
'فكان',
'فلان',
'فلس',
'فهو',
'فو',
'فوق',
'فى',
'في',
'فيفري',
'فيه',
'فيها',
'ق',
'قاطبة',
'قاف',
'قال',
'قام',
'قبل',
'قد',
'قرش',
'قطّ',
'قلما',
'قوة',
'ك',
'كأن',
'كأنّ',
'كأيّ',
'كأيّن',
'كاد',
'كاف',
'كان',
'كانت',
'كانون',
'كثيرا',
'كذا',
'كذلك',
'كرب',
'كسا',
'كل',
'كلتا',
'كلم',
'كلَّا',
'كلّما',
'كم',
'كما',
'كن',
'كى',
'كيت',
'كيف',
'كيفما',
'كِخ',
'ل',
'لأن',
'لا',
'لا سيما',
'لات',
'لازال',
'لاسيما',
'لام',
'لايزال',
'لبيك',
'لدن',
'لدى',
'لدي',
'لذلك',
'لعل',
'لعلَّ',
'لعمر',
'لقاء',
'لكن',
'لكنه',
'لكنَّ',
'للامم',
'لم',
'لما',
'لمّا',
'لن',
'له',
'لها',
'لهذا',
'لهم',
'لو',
'لوكالة',
'لولا',
'لوما',
'ليت',
'ليرة',
'ليس',
'ليسب',
'م',
'مئة',
'مئتان',
'ما',
'ما أفعله',
'ما انفك',
'ما برح',
'مائة',
'ماانفك',
'مابرح',
'مادام',
'ماذا',
'مارس',
'مازال',
'مافتئ',
'ماي',
'مايزال',
'مايو',
'متى',
'مثل',
'مذ',
'مرّة',
'مساء',
'مع',
'معاذ',
'معه',
'مقابل',
'مكانكم',
'مكانكما',
'مكانكنّ',
'مكانَك',
'مليار',
'مليم',
'مليون',
'مما',
'من',
'منذ',
'منه',
'منها',
'مه',
'مهما',
'ميم',
'ن',
'نا',
'نبَّا',
'نحن',
'نحو',
'نعم',
'نفس',
'نفسه',
'نهاية',
'نوفمبر',
'نون',
'نيسان',
'نيف',
'نَخْ',
'نَّ',
'ه',
'هؤلاء',
'ها',
'هاء',
'هاكَ',
'هبّ',
'هذا',
'هذه',
'هل',
'هللة',
'هلم',
'هلّا',
'هم',
'هما',
'همزة',
'هن',
'هنا',
'هناك',
'هنالك',
'هو',
'هي',
'هيا',
'هيهات',
'هيّا',
'هَؤلاء',
'هَاتانِ',
'هَاتَيْنِ',
'هَاتِه',
'هَاتِي',
'هَجْ',
'هَذا',
'هَذانِ',
'هَذَيْنِ',
'هَذِه',
'هَذِي',
'هَيْهات',
'و',
'و6',
'وأبو',
'وأن',
'وا',
'واحد',
'واضاف',
'واضافت',
'واكد',
'والتي',
'والذي',
'وان',
'واهاً',
'واو',
'واوضح',
'وبين',
'وثي',
'وجد',
'وراءَك',
'ورد',
'وعلى',
'وفي',
'وقال',
'وقالت',
'وقد',
'وقف',
'وكان',
'وكانت',
'ولا',
'ولايزال',
'ولكن',
'ولم',
'وله',
'وليس',
'ومع',
'ومن',
'وهب',
'وهذا',
'وهو',
'وهي',
'وَيْ',
'وُشْكَانَ',
'ى',
'ي',
'ياء',
'يفعلان',
'يفعلون',
'يكون',
'يلي',
'يمكن',
'يمين',
'ين',
'يناير',
'يوان',
'يورو',
'يوليو',
'يوم',
'يونيو',
'ّأيّان',])
def postProcessing(sums_lead3=None, sums_word_significance=None, sums_text_w2v=None):

    tmp = []
    for i in sums_text_w2v:
        if isinstance(i, list):
            tmp.append(i[0])
        else:
            tmp.append(i)
    sums_text_w2v = tmp
    
    ans = []
    if len(sums_lead3) > 0:
        ans.append(sums_lead3[0][-1])
    if sums_word_significance:
        ans.extend([sums_word_significance[0][-1]])
    if len(sums_text_w2v):
        ans.extend(sums_text_w2v)

    tmp = []
    [tmp.append(x) for x in ans if x not in tmp]
    return ''.join(tmp)


def sort_sentences(sentences,model = None,decay_rate = 1.0, pagerank_config = {'alpha': 0.85,}):

    while "ou" in sentences:
        sentences.remove("ou")

    sorted_sentences = []
    sentences_num = len(sentences)  
    graph = np.zeros((sentences_num, sentences_num))

    chapital = []
    paragraph_index = [x for x in range(sentences_num)]
    chapital.append((0,paragraph_index[0]))
    
    for i in range(len(paragraph_index)-1):
        ft = (paragraph_index[i],paragraph_index[i+1])
        chapital.append(ft)
    chapital.append((paragraph_index[-1],sentences_num))

    for x in range(sentences_num):
        sum_lie = 0
        for y in range(sentences_num):
            if x != y :
                sum_lie += graph[y,x]
        if sum_lie > 0 :
            graph [x,x] = 0
        else:
            graph [x,x] = 1.0
            sum_lie = 1.0
        for y in range(sentences_num):
            graph [y,x] = float(graph[y,x]) / sum_lie
            graph [x,y] = graph [y,x]

    for i in range(len(chapital)):
        for j in range(chapital[i][0],chapital[i][1]):
            if chapital[i][1] - chapital[i][0] <= 1 and i != len(chapital) - 1 and sentences[chapital[i][0]].find("，") < 0:
                for k in range(chapital[i+1][0],chapital[i+1][1]):
                    graph[chapital[i][0] , k] += (1.0/decay_rate - 1.0)
            elif j < chapital[i][1] - 1 and chapital[i][1] - chapital[i][0] >=3 :
                graph[j , j+1] += (1.0/decay_rate -1.0)

    for x in range(sentences_num):
        sum_lie = 0
        for y in range(sentences_num):
            sum_lie += graph[y , x]
        for y in range(sentences_num):
            graph[y,x] = float(graph[y,x]) / sum_lie
    graph = np.asnumpy(graph)
    nx_graph = nx.from_numpy_matrix(graph)
    scores = nx.pagerank(nx_graph, **pagerank_config)              # this is a dict
    sorted_scores = sorted(scores.items(), key = lambda item: item[1], reverse=True)
    np.asarray(sorted_scores)
    for index, score in sorted_scores:
        item = AttrDict(index=index, sentence=sentences[index].replace("ou","").strip(), weight=score)
        sorted_sentences.append(item)

    return sorted_sentences

class AttrDict(dict):
    """Dict that can get attribute by dot"""
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

        
class Segmentation(object):
    
    def __init__(self):
        pass

        
    def segment(self, text, lower = False):
        text = re.sub("\n","\nou",text)
        sentences = cut_sentence(text)
        words_no_stop_words = [list(filter(lambda x: x not in stop_words, sentence)) for sentence in sentences]
        return AttrDict(
                    sentences = sentences, 
                    words_no_stop_words = words_no_stop_words, 
                )
    
        
class Lead3Sum():
    def __init__(self, lead3Num):
        self.lead3Num = lead3Num

    def summarize(self, text, type_='mix'):
        """
        lead-s
        :param sentences: list
        :param type: str, you can choose 'begin', 'end' or 'mix'
        :return: list
        """
        if type(text) == str:
            self.sentences = cut_sentence(text)
        elif type(text) == list:
            self.sentences = text
        else:
            raise RuntimeError("text type must be list or str")
        if len(self.sentences) < self.lead3Num:
            return self.sentences
        num_min = min(self.lead3Num, len(self.sentences))
        if type_ == 'begin':
            summers = self.sentences[0:self.lead3Num]
        elif type_ == 'end':
            summers = self.sentences[-self.lead3Num:]
        else:
            summers = [self.sentences[0]] + [self.sentences[-1]] + self.sentences[1:self.lead3Num-1]
        summers_s = {}
        for i in range(len(summers)): # 得分计算
            if len(summers) - i == 1:
                summers_s[summers[i]] = (self.lead3Num - 0.75) / (self.lead3Num + 1)
            else:
                summers_s[summers[i]] = (self.lead3Num - i - 0.5) / (self.lead3Num + 1)
        score_sen = [(rc[1], rc[0]) for rc in sorted(summers_s.items(), key=lambda d: d[1], reverse=True)][0:num_min]
        return score_sen

class WordSignificanceSum():
    """
    Word Significance Summary
    """
    def __init__(self, word_significance , no_stop_words):
        self.no_stop_words = no_stop_words
        self.word_significance = word_significance
        self.stop_words = stop_words
        self.num = 0

    def summarize(self, text , title):

        if type(text) == str:
            self.sentences = cut_sentence(text)
        elif type(text) == list:
            self.sentences = text
        else:
            raise RuntimeError("text type must be list or str")

        for sentence in self.sentences:
            self.sentences_cut = [[word for word in tokenizer(sentence , self.no_stop_words , self.stop_words )
                          if word.strip()] for sentence in self.sentences]
        title_word = tokenizer(title , self.no_stop_words , self.stop_words)

        num_min = min(self.word_significance, len(self.sentences))
        res_sentence = []
        for word in title_word:
            for i in range(0, len(self.sentences)):
                if len(res_sentence) < num_min:
                    added = False
                    for sent in res_sentence:
                        if sent == self.sentences[i]: added = True
                    if (added == False and word in self.sentences[i]):
                        res_sentence.append(self.sentences[i])
                        break
        res_sentence = [(1-1/(len(self.sentences)+1), rs) for rs in res_sentence]
        return res_sentence

class textRankWeighted(object):
    
    def __init__(self, w2v_model):
        self.seg = Segmentation()

        self.text_cut = True
        self.sentences = None
        self.words_no_stop_words = None    
        self.key_sentences = None
        self.w2v_model = w2v_model


    def summarize(self, text, decay_rate=0.9, lower = False, 
              source = 'no_stop_words', 
              pagerank_config = {'alpha': 0.85,}):

        model = self.w2v_model
        self.key_sentences = []
        result = self.seg.segment(text=text, lower=lower)  
        self.sentences = result.sentences
        if self.text_cut:
            if len(self.sentences) > 50:
                mm = len(self.sentences) - 25
                self.sentences = result.sentences[0:25] + result.sentences[mm:]
        self.words_no_stop_words = result.words_no_stop_words

        self.key_sentences = sort_sentences(sentences = self.sentences,
                                                 model = model,
                                                 decay_rate = decay_rate,
                                                 pagerank_config = pagerank_config)

        num = len(self.key_sentences)/4
        zhaiyao_num = int(len(self.key_sentences)/10) if len(self.key_sentences)/10 >= 1 else 1
        sentence_min_len = 6
        answer = []
        count = 0
        for item in self.key_sentences:
            if count >= num:
                break
            if len(item['sentence']) >= sentence_min_len:
                answer.append(item)
                count += 1
        lastAns = []
        count_sentens = 0
        for item in sorted(answer, key = lambda x:x.index, reverse = False):
            if float(item.weight) >= 1.0/num and count_sentens <= zhaiyao_num:
                count_sentens += 1
                lastAns.append(item.sentence) #type(item.sentence)
            else:
                break

        if count_sentens == 0:
            for item in sorted(answer, key = lambda x:x.index,reverse = False):
                lastAns.append(item.sentence)
        
        tmp = []
        for i in lastAns:
            if isinstance(i, list):
                tmp.append(i[0])
            else:
                tmp.append(i)
        return tmp

class SummaryInference():
    """
    Summary inference model
    """
    def __init__(self, w2v_model_path, lead3Num = 3, word_significance = 3, no_stop_words= True , device="cpu"):
        self.w2v_model_path = w2v_model_path
        self.lead3Num = lead3Num
        self.word_significance = word_significance
        self.w2v_model = KeyedVectors.load_word2vec_format(self.w2v_model_path, binary=False)
        self.text_rank_w2v = textRankWeighted(self.w2v_model)
        self.lead3 = Lead3Sum(self.lead3Num)
        self.word_significance = WordSignificanceSum(self.word_significance , no_stop_words)

    def _inference(self, text:str , title=None) -> list:
        sentences_ = cut_sentence(text)
        sentences = [line for line in sentences_ if len(line) > 5]

        summ_lead3 = self.lead3.summarize(sentences)
        
        summ_text_w2v = self.text_rank_w2v.summarize(text)
        if title:
            summ_word_sig = self.word_significance.summarize(sentences , title)
            
            ans = postProcessing(sums_lead3=summ_lead3, sums_word_significance=summ_word_sig, sums_text_w2v=summ_text_w2v)
        else:
            ans = postProcessing(sums_lead3=summ_lead3, sums_text_w2v=summ_text_w2v)
        return ans

    def inference(self, contents, print_msg=True):
        if isinstance(contents, dict):
            summary = self._inference(contents["text"] , contents.get("title",None))
            result = {
                "text": contents["text"],
                "summary": summary
            }
            if print_msg is True:
                print(result)
            return [result]
        elif isinstance(contents, list):
            result = [{
                "text": content["text"],
                "summary": self._inference(content["text"] , content.get("title",None))
            } for content in contents]
            if print_msg is True:
                #print(result)
                pass
            return result
        else:
            print("Input format error")


