from transformers import AutoModelForQuestionAnswering, AutoTokenizer, AutoConfig
import torch
import boto3
import os
import tarfile
import io
import base64
import json
import re

s3 = boto3.client('s3')


class ServerlessModel:
    def __init__(self, model_path=None, s3_bucket=None, file_prefix=None):
        self.model, self.tokenizer = self.from_pretrained(
            model_path, s3_bucket, file_prefix)

    def from_pretrained(self, model_path: str, s3_bucket: str, file_prefix: str):
        model = self.load_model_from_s3(model_path, s3_bucket, file_prefix)
        tokenizer = self.load_tokenizer(model_path)
        return model, tokenizer

    def load_model_from_s3(self, model_path: str, s3_bucket: str, file_prefix: str):
        if model_path and s3_bucket and file_prefix:
            obj = s3.get_object(Bucket=s3_bucket, Key=file_prefix)
            bytestream = io.BytesIO(obj['Body'].read())
            tar = tarfile.open(fileobj=bytestream, mode="r:gz")
            config = AutoConfig.from_pretrained(f'{model_path}/config.json')
            for member in tar.getmembers():
                if member.name.endswith(".bin"):
                    f = tar.extractfile(member)
                    state = torch.load(io.BytesIO(f.read()))
                    model = AutoModelForQuestionAnswering.from_pretrained(
                        pretrained_model_name_or_path=None, state_dict=state, config=config)
            return model
        else:
            raise KeyError('No S3 Bucket and Key Prefix provided')

    def load_tokenizer(self, model_path: str):
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        return tokenizer

    def encode(self, question, context):
        encoded = self.tokenizer.encode_plus(question, context)
        return encoded["input_ids"], encoded["attention_mask"]

    def decode(self, token):
        answer_tokens = self.tokenizer.convert_ids_to_tokens(
            token, skip_special_tokens=True)
        return self.tokenizer.convert_tokens_to_string(answer_tokens)

    def predict(self, question, context):
        input_ids, attention_mask = self.encode(question, context)
        start_scores, end_scores = self.model(torch.tensor(
            [input_ids]), attention_mask=torch.tensor([attention_mask]))
        ans_tokens = input_ids[torch.argmax(
            start_scores): torch.argmax(end_scores)+1]
        answer = self.decode(ans_tokens)
        return answer
