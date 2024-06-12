import torch
import transformers
import time
from transformers import LlamaForCausalLM, LlamaTokenizer, TextStreamer
import re
import urllib.request
import time
from statistics import mean
import sys
import timeit
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
    inputs = tokenizer(prompt, return_tensors="pt")
    return inputs


def inference(model,tokenized_inputs):
    generate_ids, timings = model.generate(tokenized_inputs.input_ids, max_length=50, num_beams=4)
    return generate_ids, timings

def postprocess(tokenizer,generated_ids):
    output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return output

if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Input words is not specified by default taking as 1000")
        sys.argv.append("1000")
    elif len(sys.argv) == 2 and int(sys.argv[1]) > 40000:
        print("Input words should be less than 40000, so running for max limit 40000")
        sys.argv[1] = ("40000")
    model = load_model()
    tokenizer = load_tokenizer()
    gen_token = []
    gen_words = []
    pre_time = []
    pre_total = []
    WT_ratio = []
    TPS = []
    pro_time = []
    pro_total = []
    TW_ratio = []
    WPS = []
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
        # tokenization_start_time = time.time()

        pool = Pool(thread_count)
        ip_args_list = []
        for i in range(thread_count):
            ip_args_list.append((tokenizer,prompt))
        tokenization_start_time = timeit.default_timer()
        tokenized_inputs = pool.starmap(preprocess, ip_args_list)
        # tokenization_end_time = time.time()
        tokenization_end_time = timeit.default_timer()
        pool.close()
        pool.join()



        # postprocess_start_time = time.time()
        postprocess_start_time = timeit.default_timer()
        pool = Pool(thread_count)
        op_args_list = []
        gen_token_per_cycle = []
        for i in range(thread_count):
            op_args_list.append((tokenizer,tokenized_inputs[i]["input_ids"]))
            no_tokens = tokenized_inputs[i]["input_ids"].shape[1]
            gen_token_per_cycle.append(no_tokens)
        # postprocess_start_time = time.time()
        postprocess_start_time = timeit.default_timer()
        postprocessed_output = pool.starmap(postprocess, op_args_list)
        # postprocess_end_time = time.time()
        postprocess_end_time = timeit.default_timer()
        pool.close()
        pool.join()
        gen_word_per_cycle = []
        for i in range(thread_count):
            op_list = postprocessed_output[i].split(" ")
            no_of_words = len(op_list)
            gen_word_per_cycle.append(no_of_words)
        # op_list = postprocessed_output.split(" ")
        # no_of_words = len(op_list)
        pre_time.append(((tokenization_end_time-tokenization_start_time)*1000)/(ip_words*thread_count))
        pre_total.append(((tokenization_end_time-tokenization_start_time)*1000))
        print("pre_time: ", pre_time)
        print("ip_words*thread_count: ", ip_words*thread_count)
        print("(tokenization_end_time-tokenization_start_time): ", (tokenization_end_time-tokenization_start_time))
        
        WT_ratio.append(mean(gen_word_per_cycle)/mean(gen_token_per_cycle))
        TPS.append(mean(gen_token_per_cycle)/(tokenization_end_time-tokenization_start_time))

        gen_token.append(mean(gen_token_per_cycle))
        gen_words.append(mean(gen_word_per_cycle))
        pro_time.append(((postprocess_end_time-postprocess_start_time)*1000)/mean(gen_token_per_cycle))
        pro_total.append((postprocess_end_time-postprocess_start_time)*1000)
        # print("pro_time: ", pro_time)
        TW_ratio.append(mean(gen_token_per_cycle)/mean(gen_word_per_cycle))
        WPS.append(mean(gen_word_per_cycle)/(postprocess_end_time-postprocess_start_time))
    print("\n")
    print("input words: ", ip_words*thread_count, "\n")
    print("Preprocess time per token: ", mean(pre_time))
    print("Total pretime ; ", mean(pre_total))
    print("TPS: ", mean(TPS))
    print("W:T ratio: ", mean(WT_ratio))
    print("\n")
    print("ip token to post proc: ", mean(gen_token))
    print("Postprocess time per token: ", mean(pro_time))
    print("Total postprocess time: ", mean(pro_total))
    print("WPS: ", mean(WPS))
    print("T:W ratio: ", mean(TW_ratio))
    print("\n")