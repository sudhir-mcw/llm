import torch
import transformers
import time
from transformers import LlamaForCausalLM, LlamaTokenizer, TextStreamer
import psutil
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
PROFILE = True

'''
    profiling function uses torch.prolfer.profile(),
    logs to a file name with functions name  
    usage : 
    @profile
    def fun_to_profile():
        pass
'''
import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"
def profile(func):
    def profiler(*args, **kwargs):
        if not PROFILE:
            return func(*args, **kwargs)
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU],
            on_trace_ready=torch.profiler.tensorboard_trace_handler(f'./exp-logs/{func.__name__}'),
            profile_memory=True,
            with_stack=True
            ) as prof:
            generated_ids = func(*args, **kwargs)
        return generated_ids
 
    return profiler

def load_model(model_dir=None):
    if not model_dir:
        model_dir  = "/DATA1/olaisw/TinyLlama-1.1B-Chat-v0.6"
    model = LlamaForCausalLM.from_pretrained(model_dir)  #### Loading checkpoint shards:---> starts here
    return model
def load_tokenizer(model_dir=None):
    if not model_dir:
        model_dir  = "/DATA1/olaisw/TinyLlama-1.1B-Chat-v0.6"
    tokenizer = LlamaTokenizer.from_pretrained(model_dir)
    return tokenizer

@profile
def preprocess(tokenizer,prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    return inputs


def inference(model,tokenized_inputs):
    generate_ids, timings = model.generate(tokenized_inputs.input_ids, max_length=50, num_beams=4)
    return generate_ids, timings

@profile
def postprocess(tokenizer,generated_ids):
    output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
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

    count   =   0
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
def chunkify(lst,n):
    return [lst[i::n] for i in range(1, n+1)]
if __name__ == "__main__":
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

    tokenizer = load_tokenizer()
    gen_token = []
    gen_words = []
    n_cycle = 3
    thread_count = 4

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
    preprocess_time, postprocess_time = profiler()
    tps = round(mean(gen_token)/(preprocess_time/1000), 2)
    wps = round(mean(gen_words)/(postprocess_time/1000), 2)
    print("\n Preprocess time per token: ", round(preprocess_time/mean(gen_token), 4))
    print("Tokens per sec: ", tps)
    print("\nPost process time per token: ", round(postprocess_time/mean(gen_words), 4))
    print("Words per sec: ", wps)
    print("\n")