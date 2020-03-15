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
