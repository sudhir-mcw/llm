import torch
import transformers
import time
from transformers import LlamaForCausalLM, LlamaTokenizer, TextStreamer
# import psutil
import re
import urllib.request
import time
from statistics import mean
import shutil
import os
import json
import codecs
import sys
from memory_profiler import profile
from guppy import hpy
# from filprofiler.api import profile
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

@profile(precision=4)
def preprocess(tokenizer,prompt):
    # inputs = profile(lambda: tokenizer(prompt, return_tensors="pt"), "./tmp/fil-result_preprocess")
    inputs = tokenizer(prompt, return_tensors="pt")
    return inputs


def inference(model,tokenized_inputs):
    generate_ids, timings = model.generate(tokenized_inputs.input_ids, max_length=50, num_beams=4)
    return generate_ids, timings

# @profile
def postprocess(tokenizer,generated_ids):
    output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return output

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
    heap_memory = []

    url='https://raw.githubusercontent.com/mxw/grmr/master/src/finaltests/bible.txt'
    doc=urllib.request.urlopen(url).read().decode('utf-8')
    regex = r'(\w*) '
    words_list = re.findall(regex,doc)
    # prompt = "Give me some places to visit on vacation?"
    prompt_str = ' '
    ip_words = int(sys.argv[1])
    n_cycle = 2
    for i in range(0, n_cycle):
        prompt = prompt_str.join(words_list[:ip_words])
        # print(type(h.heap()))
        # h.setrelheap()
        # print("before heap: ", h.heap())
        tokenized_inputs = preprocess(tokenizer, prompt)
        # heapinfo = h.heap()
        # # print(heapinfo.bytype)
        # byrcs = heapinfo.byrcs
        # print("heap: ", heapinfo)
        # print(byrcs)
        # # i = iter(byrcs.nodes)
        # # for val in i:
        # a = re.search(r'\b(Total size = )\b', str(byrcs))
        # byt = re.search(r'\b( bytes)\b', str(byrcs))
        # print(a.start())
        # print(a.end())
        # print(byt.start())
        # # print("val: ", str(val))
        # total_size = str(byrcs)[a.end():byt.start()]
        # print("total size = ", int(total_size.replace(" ", "")))
        # heap_memory.append(int(total_size.replace(" ", "")))
        # print(byrcs[4].byclodo)
        # print(byrcs[4].byid)
        # print("\n byrcs[4].byvia: ", byrcs[4].byvia)
        no_tokens = tokenized_inputs["input_ids"].shape[1]
        gen_token.append(no_tokens)
        postprocessed_output = postprocess(tokenizer,tokenized_inputs['input_ids'])
        op_list = postprocessed_output.split(" ")
        no_of_words = len(op_list)
        gen_words.append(no_of_words)
    time.sleep(5)
    print("\n")
    print("input words: ", ip_words,)
    print("ip token to post proc: ", mean(gen_token))
    print("op words: ", mean(gen_words))
    # print("Memory 1st iter: ", heap_memory[0])
    # print("Memory avg of remaining: ", mean(heap_memory[1:]))
    # preprocess_time, postprocess_time = profiler()
    # wt_ratio = round(ip_words/mean(gen_token), 3)
    # tps = round(mean(gen_token)/(preprocess_time/1000), 2)
    # tw_ratio = round(mean(gen_token)/mean(gen_words), 3)
    # wps = round(mean(gen_words)/(postprocess_time/1000), 2)
    # print("\n Preprocess time per token: ", round(preprocess_time/mean(gen_token), 4))
    # print("WT ratio: ", wt_ratio)
    # print("Tokens per sec: ", tps)
    # print("\nPost process time per token: ", round(postprocess_time/mean(gen_words), 4))
    # print("TW ratio: ", tw_ratio)
    # print("Words per sec: ", wps)
    # print("\n")