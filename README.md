# Examples

### Document classification
```
from BERT import DcInference
model = DcInference(model_path='models/dc', device='cpu')
model.inference("Sinovation Ventures AI Institute is located in Shenzhen ...")
#label
model.inference(["Sinovation Ventures AI Institute is located in Shenzhen ..."])
#[{"text","Sinovation Ventures AI Institute is located in Shenzhen ...","label":"Tech"}]
```

### Name entity recognition
```
from BERT import NerInference
model = NerInference(model_path='models/ner', device='cpu')
model.inference("Sinovation Ventures AI Institute is located in Shenzhen ...")
#label
model.inference(["Sinovation Ventures AI Institute is located in Shenzhen ..."])
#[{"text","Sinovation Ventures AI Institute is located in Shenzhen ...","label":[{"entity":"Sinovation Ventures AI Institute","type":"ORG","position":0},{"entity":"Shenzhen","type":"LOC","position":8}]}]
```

### Sentiment analysis
```
from BERT import SaInference
model = SaInference(model_path='models/sa', device='cpu')
model.inference("Sinovation Ventures AI Institute is located in Shenzhen ...")
#label
model.inference(["Sinovation Ventures AI Institute is located in Shenzhen ..."])
#[{"text","Sinovation Ventures AI Institute is located in Shenzhen ...","label":"Neutral"}]
```

### Sentence similarity computation
```
from BERT import StsInference
model = StsInference(model_path='models/sts', device='cpu')
model.inference(text_a="Sinovation Ventures AI Institute is located in Shenzhen ...",text_b="Sinovation Ventures AI Institute, which is located in Shenzhen ...")
#label
model.inference(texts=[["Sinovation Ventures AI Institute is located in Shenzhen ...","Sinovation Ventures AI Institute, which is located in Shenzhen ..."]])
#[{"text":["Sinovation Ventures AI Institute is located in Shenzhen ...","Sinovation Ventures AI Institute, which is located in Shenzhen ..."],"label":"1"}]
```
