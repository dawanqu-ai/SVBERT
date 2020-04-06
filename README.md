- DC 
text：وفاة الممثلة الإيطالية فِيرْنا ليسي عن عمر ناهز 78 عاما  الممثلة الإيطالية الشقراء فيرْنا ليسي تفارق الحياة عن عمر ناهز ثمانية وسبعين عاما بعد مشوار فني ثري وطويل في كل من إيطاليا وفرنسا والولايات المتحدة الأمريكية إلى جانب كبار المخرجين العالميين. فيرنا ليسي توفيتْ خلال نومها بعد معاناة طويلة من المرض.

label：Culture


demo：
```
from BERT import DcInference
model = DcInference(model_path='models/dc', device='cpu')
model.inference("وفاة الممثلة الإيطالية فِيرْنا ليسي عن عمر ناهز 78 عاما  المم ...")
#label
model.inference(["وفاة الممثلة الإيطالية فِيرْنا ليسي عن عمر ناهز 78 عاما  المم ..."])
#[{"text","وفاة الممثلة الإيطالية فِيرْنا ليسي عن عمر ناهز 78 عاما  المم ...","label":"Culture"}]

```

- NER
text and label:
كما	O
ان	O
شركتي	O
تويوتا	B-ORG
وجنرال	B-ORG
موتورز	I-ORG
زادتا	O
من	O
حجم	O
استثماراتهما	O
في	O
المشروعات	O
المشتركة	O
في	O
الصين	B-LOC
.	O


```
from BERT import NerInference
model = NerInference(model_path='models/ner', device='cpu')
model.inference("كما ان شركتي تويوتا وجنرال موتورز زادتا من حجم استثماراتهما في المشروعات المشتركة في الصين.")
#label
model.inference([كما ان شركتي تويوتا وجنرال موتورز زادتا من حجم استثماراتهما في المشروعات المشتركة في الصين."])
#[{"text","كما ان شركتي تويوتا وجنرال موتورز زادتا من حجم استثماراتهما في المشروعات المشتركة في الصين.","label":[{"entity":"تويوتا وجنرال موتورز","type":"ORG","position":3},{"entity":"الصين","type":"LOC","position":14}]}]
```

- SA

text: مباراة اليوم | ذهاب تصفيات كاس العالم 2018 لقارة اوروبا السويد x ايطاليا الساعة 10:45 القناة:أبوظبي HD

label : Neutral

```
from BERT import SaInference
model = SaInference(model_path='models/sa', device='cpu')
model.inference("مباراة اليوم | ذهاب تصفيات كاس العالم 2018 لقارة اوروبا السويد x ايطاليا الساعة 10:45 القناة:أبوظبي HD")
#label
model.inference(["مباراة اليوم | ذهاب تصفيات كاس العالم 2018 لقارة اوروبا السويد x ايطاليا الساعة 10:45 القناة:أبوظبي HD"])
#[{"text","مباراة اليوم | ذهاب تصفيات كاس العالم 2018 لقارة اوروبا السويد x ايطاليا الساعة 10:45 القناة:أبوظبي HD","label":"Neutral"}]
```


- STS

text_a: ماذا نقصد بالباستا فلورا؟
text_b: ما تعريف الباستا فلورا؟
label: 1


```
from BERT import StsInference
model = StsInference(model_path='models/sts', device='cpu')
model.inference(text_a="ماذا نقصد بالباستا فلورا؟",text_b="ما تعريف الباستا فلورا؟")
#label
model.inference(texts=[["ماذا نقصد بالباستا فلورا؟","ما تعريف الباستا فلورا؟"]])
#[{"text":["ماذا نقصد بالباستا فلورا؟","ما تعريف الباستا فلورا؟"],"label":"1"}]
```