from transformers import AutoModelForSequenceClassification, AutoTokenizer
from FlagEmbedding import FlagReranker
from configs import RERANK_MODEL, MODEL_PATH
from server.utils import rerank_device
import os


class ReRankModel:
    def __init__(self, model=RERANK_MODEL, device=rerank_device()):
        self.model = model
        self.path = MODEL_PATH["rerank_model"][model]
        self.device = device
        self.rerank_model = None
        # self.rerank_tokenizer = None

    def _load_reranks(self, model: str = RERANK_MODEL, device: str = rerank_device()):
        # if self.rerank_model is not None and self.rerank_tokenizer is not None:
        #     return
        # self.rerank_tokenizer = AutoTokenizer.from_pretrained(self.path)
        # self.model = AutoModelForSequenceClassification.from_pretrained(self.path)
        # self.model.eval()
        # 设置可见的 GPU 号
        if self.rerank_model is None:
            self.rerank_model = FlagReranker(self.path,
                                             use_fp16=True)  # Setting use_fp16 to True speeds up computation with a slight performance degradation

    def rerank(self, pairs, top_k):
        self._load_reranks()
        # with torch.no_grad():
        #     inputs = self.rerank_tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
        #     scores = self.model(**inputs, return_dict=True).logits.view(-1, ).float()
        scores = self.rerank_model.compute_score(pairs)
        sorted_list = sorted(zip([ans[1] for ans in pairs], scores), key=lambda x: -x[1])
        return sorted_list[:top_k]


modelPool = {}


def load_rerank_model(model: str, device: str):
    if str not in modelPool.keys():
        modelPool[str] = ReRankModel(model, device)
    return modelPool[str]


if __name__ == "__main__":
    pairs = [['what is panda?', 'hi'], ['what is panda?',
                                        'The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.']]
    print(ReRankModel().rerank(pairs, 2))
