# Examples

### Document classification

```
from BERT import DcInference
model = DcInference(model_path='models/dc', device='cpu')
model.inference("وفاة الممثلة الإيطالية فِيرْنا ليسي عن عمر ناهز 78 عاما  المم ...")
#label
model.inference(["وفاة الممثلة الإيطالية فِيرْنا ليسي عن عمر ناهز 78 عاما  المم ..."])
#[{"text","وفاة الممثلة الإيطالية فِيرْنا ليسي عن عمر ناهز 78 عاما  المم ...","label":"Culture","score":{"Culture":1.0}]

```

### Name entity recognition

```
from BERT import NerInference
model = NerInference(model_path='models/ner', device='cpu')
model.inference("كما ان شركتي تويوتا وجنرال موتورز زادتا من حجم استثماراتهما في المشروعات المشتركة في الصين.")
#label
model.inference([كما ان شركتي تويوتا وجنرال موتورز زادتا من حجم استثماراتهما في المشروعات المشتركة في الصين."])
#[{"text","كما ان شركتي تويوتا وجنرال موتورز زادتا من حجم استثماراتهما في المشروعات المشتركة في الصين.","label":[{"entity":"تويوتا وجنرال موتورز","type":"ORG","position":3},{"entity":"الصين","type":"LOC","position":14}]}]
```

### Sentiment analysis

```
from BERT import SaInference
model = SaInference(model_path='models/sa', device='cpu')
model.inference("مباراة اليوم | ذهاب تصفيات كاس العالم 2018 لقارة اوروبا السويد x ايطاليا الساعة 10:45 القناة:أبوظبي HD")
#label
model.inference(["مباراة اليوم | ذهاب تصفيات كاس العالم 2018 لقارة اوروبا السويد x ايطاليا الساعة 10:45 القناة:أبوظبي HD"])
#[{"text","مباراة اليوم | ذهاب تصفيات كاس العالم 2018 لقارة اوروبا السويد x ايطاليا الساعة 10:45 القناة:أبوظبي HD","label":"Neutral","score":{"Neutral":0.8,"Positive":0.1,"Negative":0.1}]
```


### Sentence similarity computation

```
from BERT import StsInference
model = StsInference(model_path='models/sts', device='cpu')
model.inference(text_a="ماذا نقصد بالباستا فلورا؟",text_b="ما تعريف الباستا فلورا؟")
#label
model.inference(texts=[["ماذا نقصد بالباستا فلورا؟","ما تعريف الباستا فلورا؟"]])
#[{"text":["ماذا نقصد بالباستا فلورا؟","ما تعريف الباستا فلورا؟"],"label":"1","score":{"1":0.9,"0":0.1}]
```