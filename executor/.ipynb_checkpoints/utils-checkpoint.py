from transformers import BertTokenizer, RobertaTokenizer


def build_tokenizer(cfg):
    if cfg.MODEL.BERT_TYPE == "BERT":
        return BertTokenizer.from_pretrained("bert-base-uncased")
    elif cfg.MODEL.BERT_TYPE == "ROBERTA":
        return RobertaTokenizer.from_pretrained(cfg.MODEL.BERT_NAME)
    assert False
