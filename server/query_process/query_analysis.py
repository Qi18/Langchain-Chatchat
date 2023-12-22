import torch
from fastapi import Body
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForTokenClassification, pipeline
from configs import IR_MODEL, MODEL_PATH, NER_MODEL

ir_modelPool = {}
ner_modelPool = {}


class IRModel:
    def __init__(self, model=IR_MODEL, use_fp16=False):
        self.model = model
        self.path = MODEL_PATH["ir_model"][self.model]

        self.tokenizer = AutoTokenizer.from_pretrained(self.path)
        self.ir_model = AutoModelForSequenceClassification.from_pretrained(self.path)
        if use_fp16:
            self.ir_model.half()
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        # self.device = torch.device("cuda:1")
        self.ir_model = self.ir_model.to(self.device)
        self.ir_model.eval()

    def query_intent_recognition(self, query: str):
        assert isinstance(query, str)

        # 对查询进行编码
        encoded_inputs = self.tokenizer(query, padding=True, truncation=True, return_tensors='pt', max_length=128)

        encoded_inputs = encoded_inputs.to(self.device)

        # 使用BERT模型进行预测
        with torch.no_grad():
            outputs = self.ir_model(**encoded_inputs)

        # 获取预测结果
        predictions = torch.argmax(outputs.logits, dim=-1).tolist()

        predicted_labels = [self.ir_model.config.id2label[id] for id in predictions]

        return predicted_labels[0]


class NerModel:
    def __init__(self, model=NER_MODEL, use_fp16=False):
        self.model = model
        self.path = MODEL_PATH["ner_model"][self.model]

        self.tokenizer = AutoTokenizer.from_pretrained(self.path, model_max_length=256)
        # self.ner_model = AutoModelForTokenClassification.from_pretrained(self.path)
        # if use_fp16:
        #     self.ner_model.half()
        # self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        # self.ner_model = self.ner_model.to(self.device)
        # self.ner_model.eval()
        self.pipeline = pipeline("ner", model=self.path, tokenizer=self.tokenizer)

    def ner(self, query: str):
        assert isinstance(query, str)

        res = self.pipeline(query)
        # print(res)
        merged_entities = []
        current_entity = {}

        for entity in res:
            entity_type = entity['entity'].split('-')[1]

            if 'B-' in entity['entity']:
                if current_entity:
                    merged_entities.append(current_entity)
                current_entity = {'entity': entity_type, 'start': entity['start'], 'end': entity['end']}
            elif 'I-' in entity['entity'] or 'E-' in entity['entity']:
                current_entity['end'] = entity['end']

        if current_entity:
            merged_entities.append(current_entity)

        for item in merged_entities:
            item['words'] = query[item['start']: item['end']]

        return merged_entities


def load_intent_recognition_model(model: str = IR_MODEL):
    if model not in ir_modelPool.keys():
        # print(f"load rerank model {model}")
        ir_modelPool[model] = IRModel(model)
    return ir_modelPool[model]


def load_ner_model(model: str = NER_MODEL):
    if model not in ner_modelPool.keys():
        # print(f"load rerank model {model}")
        ner_modelPool[model] = NerModel(model)
    return ner_modelPool[model]

def query_ir(query: str):
    ir_model = load_intent_recognition_model()
    intent = ir_model.query_intent_recognition(query)
    return intent


def query_ner(query: str):
    ner_model = load_ner_model()
    entities = ner_model.ner(query)
    return entities

def ir_query(query: str = Body(..., description="用户输入", examples=["你好"]),):
    return query_ir(query)


if __name__ == "__main__":
    query = "我想看看今天的新闻"
    print(query_ir(query))
    print(query_ner(query))
