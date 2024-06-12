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
from multiprocess import Pool

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
    mlti_thread = False
    process = psutil.Process()
    if len(sys.argv) == 1:
        print("Input words is not specified by defaut taking as 1000")
        sys.argv.append("1000")
    elif len(sys.argv) == 2 and int(sys.argv[1]) > 40000:
        print("Input words exceeds max limit, so limiting to 40000")
        sys.argv[1] = "40000"

    time.sleep(1)

    model = load_model()
    tokenizer = load_tokenizer()
    gen_token = []
    gen_words = []
    pre_user_time = []
    pre_system_time = []
    post_user_time = []
    post_system_time = []
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
        cpu_times_before = process.cpu_times()
        tokenized_inputs = pool.starmap(preprocess, ip_args_list)
        cpu_times_after = process.cpu_times()
        
        pool.close()
        pool.join()
        pre_user_time.append(cpu_times_after.user - cpu_times_before.user)
        pre_system_time.append(cpu_times_after.system - cpu_times_before.system)

        pool = Pool(thread_count)
        op_args_list = []
        for i in range(thread_count):
            op_args_list.append((tokenizer,tokenized_inputs[i]["input_ids"]))
            no_tokens = tokenized_inputs[i]["input_ids"].shape[1]
            gen_token.append(no_tokens)
        
        cpu_times_before = process.cpu_times()
        postprocessed_output = pool.starmap(postprocess, op_args_list)
        cpu_times_after = process.cpu_times()
        pool.close()
        pool.join()
        for i in range(thread_count):
            op_list = postprocessed_output[i].split(" ")
            no_of_words = len(op_list)
            gen_words.append(no_of_words)




        # no_tokens = tokenized_inputs["input_ids"].shape[1]
        # gen_token.append(no_tokens)

        post_user_time.append(cpu_times_after.user - cpu_times_before.user)
        post_system_time.append(cpu_times_after.system - cpu_times_before.system)

        # op_list = postprocessed_output.split(" ")
        # no_of_words = len(op_list)
        # gen_words.append(no_of_words)
    
    cpu_freq = psutil.cpu_freq().max

    pre_cpu_cycles_user = mean(pre_user_time) * cpu_freq * 1e6
    pre_cpu_cycles_system = mean(pre_system_time) * cpu_freq * 1e6

    post_cpu_cycles_user = mean(post_user_time) * cpu_freq * 1e6
    post_cpu_cycles_system = mean(post_system_time) * cpu_freq * 1e6

    print("cpu_freq :", cpu_freq)
    print("pre_user_time :", pre_user_time)
    print("post_user_time :", post_user_time)
    print("ip_words :", ip_words*thread_count)

    print(f"\nPreProcess User CPU time: {mean(pre_user_time):.6f} seconds")
    print(f"PreProcess System CPU time: {mean(pre_system_time):.6f} seconds")
    print(f"PreProcess CPU cycles (user): {pre_cpu_cycles_user:.0f}")
    print(f"PreProcess CPU cycles (system): {pre_cpu_cycles_system:.0f}")
    print(f"Total PreProcess CPU cycles: {pre_cpu_cycles_system + pre_cpu_cycles_user:.0f}")

    print(f"\nPostProcess User CPU time: {mean(post_user_time):.6f} seconds")
    print(f"PostProcess System CPU time: {mean(post_system_time):.6f} seconds")
    print(f"PostProcess CPU cycles (user): {post_cpu_cycles_user:.0f}")
    print(f"PostProcess CPU cycles (system): {post_cpu_cycles_system:.0f}")
    print(f"Total PostProcess CPU cycles: {post_cpu_cycles_system + post_cpu_cycles_user:.0f}")
    