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
from multiprocess import Pool
from filprofiler.api import profile
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
    inputs = profile(lambda: tokenizer(prompt, return_tensors="pt"), "./tmp/fil-result_preprocess")
    # inputs = tokenizer(prompt, return_tensors="pt")
    return inputs


def inference(model,tokenized_inputs):
    generate_ids, timings = model.generate(tokenized_inputs.input_ids, max_length=50, num_beams=4)
    return generate_ids, timings

def postprocess(tokenizer,generated_ids):
    output = profile(lambda: tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0], "./tmp/fil-result_postprocess")
    # output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return output[0]

def profiler():
    pre_process_cpu_times = []
    post_process_cpu_times = []

    count = 0
    for file in os.listdir(os.path.join(os.getcwd(),"exp-logs/preprocess")):
        file = os.path.join(os.getcwd(),"exp-logs/preprocess",file)
        with codecs.open(file, 'r', encoding='utf-8',
                 errors='replace') as fdata:
            json_data = json.load(fdata, strict=False)
        
        trace_events = json_data["traceEvents"]
        entry_count = 0
        for event in trace_events:
            if " preprocess" in event.get("name"):
                pre_process_cpu_times.append(int(event.get("dur")))
                entry_count+=1
        count+=1

    count = 0
    for file in os.listdir(os.path.join(os.getcwd(),"exp-logs/postprocess")):

        file = os.path.join(os.getcwd(),"exp-logs/postprocess",file)
        with codecs.open(file, 'r', encoding='utf-8',
                 errors='replace') as fdata:
            json_data = json.load(fdata, strict=False)
        
        trace_events = json_data["traceEvents"]
        entry_count = 0
        for event in trace_events:
            if "postprocess" in event.get("name"):
                print("\n post each: ", int(event.get("dur")))
                post_process_cpu_times.append(int(event.get("dur")))
                entry_count+=1

        count+=1

    return mean(pre_process_cpu_times)/1000, mean(post_process_cpu_times)/1000

if __name__ == "__main__":
    h = hpy()
    if len(sys.argv) == 1:
        print("Input words is not specified by defaut taking as 1000")
        sys.argv.append("1000")
    elif len(sys.argv) == 2 and int(sys.argv[1]) > 40000:
        print("Input words exceeds max limit, so limiting to 40000")
        sys.argv[1] = "40000"
    if os.path.exists(r"./exp-logs/preprocess"):
        shutil.rmtree(r"./exp-logs/preprocess")
    if os.path.exists(r"./exp-logs/postprocess"):
        shutil.rmtree(r"./exp-logs/postprocess")
    time.sleep(1)

    model = load_model()
    tokenizer = load_tokenizer()
    gen_token = []
    gen_words = []
    thread_count = 4
    n_cycle = 5

    url='https://raw.githubusercontent.com/mxw/grmr/master/src/finaltests/bible.txt'
    doc=urllib.request.urlopen(url).read().decode('utf-8')
    regex = r'(\w*) '
    words_list = re.findall(regex,doc)
    # prompt = "Give me some places to visit on vacation?"
    prompt_str = ' '
    ip_words = int(sys.argv[1])

    for i in range(0, n_cycle):
        prompt = prompt_str.join(words_list[:ip_words])

        pool = Pool(thread_count)
        ip_args_list = []
        for i in range(thread_count):
            ip_args_list.append((tokenizer,prompt))
        tokenized_inputs = pool.starmap(preprocess, ip_args_list)
        pool.close()
        pool.join()

        pool = Pool(thread_count)
        op_args_list = []
        for i in range(thread_count):
            op_args_list.append((tokenizer,tokenized_inputs[i]["input_ids"]))
            no_tokens = tokenized_inputs[i]["input_ids"].shape[1]
            gen_token.append(no_tokens)
        postprocessed_output = pool.starmap(postprocess, op_args_list)
        pool.close()
        pool.join()
        for i in range(thread_count):
            op_list = postprocessed_output[i].split(" ")
            no_of_words = len(op_list)
            gen_words.append(no_of_words)
    time.sleep(2)
    print("\n")
    print("input words: ", ip_words*thread_count)
    print("ip token to post proc: ", mean(gen_token))
    print("op words: ", mean(gen_words))
