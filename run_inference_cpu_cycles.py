import torch
import transformers
import time
from transformers import LlamaForCausalLM, LlamaTokenizer, TextStreamer
import re
import urllib.request
import time
from statistics import mean
import shutil
import os
import json
import codecs
import sys
import psutil
import cProfile
import pstats
import io
import timeit
from pyperf import perf_counter , Runner
import random
PROFILE = True


def load_model(model_dir=None):
    if not model_dir:
        model_dir  = "../TinyLlama-1.1B-Chat-v0.6"
    model = LlamaForCausalLM.from_pretrained(model_dir)  #### Loading checkpoint shards:---> starts here
    return model
def load_tokenizer(model_dir=None):
    if not model_dir:
        model_dir  = "../TinyLlama-1.1B-Chat-v0.6"
    tokenizer = LlamaTokenizer.from_pretrained(model_dir)
    return tokenizer

def preprocess(tokenizer,prompt):
    # inputs = profile(lambda: tokenizer(prompt, return_tensors="pt"), "./tmp/fil-result_preprocess")
    inputs = tokenizer(prompt, return_tensors="pt")
    return inputs

def inference(model,tokenized_inputs):
    generate_ids, timings = model.generate(tokenized_inputs.input_ids, max_length=50, num_beams=4)
    return generate_ids, timings

def postprocess(tokenizer,generated_ids):
    output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return output

if __name__ == "__main__":
    ip_words = 10000 # change input words accordingly
    operation = 'preprocess' # change options to preprocess/postprocess
    process = psutil.Process()

    time.sleep(1)

    model = load_model()
    tokenizer = load_tokenizer()
    gen_token = []
    gen_words = []
    pre_user_time = []
    pre_system_time = []
    post_user_time = []
    post_system_time = []

    url='https://raw.githubusercontent.com/mxw/grmr/master/src/finaltests/bible.txt'
    doc=urllib.request.urlopen(url).read().decode('utf-8')
    regex = r'(\w*) '
    words_list = re.findall(regex,doc)
    # prompt = "Give me some places to visit on vacation?"
    prompt_str = ' '
    
    n_cycle = 1
    runner = Runner()
    runner.metadata['description'] = "DeltaBlue benchmark"
    for i in range(0, n_cycle):
        prompt = prompt_str.join(words_list[:ip_words])
        cpu_times_before = process.cpu_times()
        if operation == 'preprocess':
            runner.bench_func('preprocess', preprocess, tokenizer,prompt)
        elif operation == 'postprocess':
            tokenized_inputs = preprocess(tokenizer, prompt)
            runner.bench_func('postprocess', postprocess, tokenizer,tokenized_inputs['input_ids'])
    