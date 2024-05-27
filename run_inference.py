import torch
import transformers
import time
from transformers import LlamaForCausalLM, LlamaTokenizer, TextStreamer
import re
import urllib.request
import time
from statistics import mean
import sys

def load_model(model_dir=None):
    if not model_dir:
        model_dir  = "/home/mcwaiteam/TinyLlama-1.1B-Chat-v0.6"
    model = LlamaForCausalLM.from_pretrained(model_dir)  #### Loading checkpoint shards:---> starts here
    return model
def load_tokenizer(model_dir=None):
    if not model_dir:
        model_dir  = "/home/mcwaiteam/TinyLlama-1.1B-Chat-v0.6"
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
    elif len(sys.argv) == 2 and sys.argv[1] > 40000:
        print("Input words should be less than 40000, so running for max limit 40000")
        sys.argv[1] = ("40000")
    model = load_model()
    tokenizer = load_tokenizer()
    gen_token = []
    pre_time = []
    pre_total = []
    WT_ratio = []
    TPS = []
    pro_time = []
    pro_total = []
    TW_ratio = []
    WPS = []

    url='https://raw.githubusercontent.com/mxw/grmr/master/src/finaltests/bible.txt'
    doc=urllib.request.urlopen(url).read().decode('utf-8')
    regex = r'(\w*) '
    words_list = re.findall(regex,doc)
    # prompt = "Give me some places to visit on vacation?"
    prompt_str = ' '
    ip_words = int(sys.argv[1])

    for i in range(0, 5):
        prompt = prompt_str.join(words_list[:ip_words])
        tokenization_start_time = time.time()
        tokenized_inputs = preprocess(tokenizer, prompt)
        tokenization_end_time = time.time()
        pre_time.append(((tokenization_end_time-tokenization_start_time)*1000)/ip_words)
        pre_total.append(((tokenization_end_time-tokenization_start_time)*1000))
        no_tokens = tokenized_inputs["input_ids"].shape[1]
        gen_token.append(no_tokens)
        WT_ratio.append(ip_words/no_tokens)
        TPS.append(no_tokens/(tokenization_end_time-tokenization_start_time))

        postprocess_start_time = time.time()
        postprocessed_output = postprocess(tokenizer,tokenized_inputs['input_ids'])

        postprocess_end_time = time.time()
        op_list = postprocessed_output.split(" ")
        no_of_words = len(op_list)
        pro_time.append(((postprocess_end_time-postprocess_start_time)*1000)/no_tokens)
        pro_total.append((postprocess_end_time-postprocess_start_time)*1000)
        
        TW_ratio.append(no_tokens/no_of_words)
        WPS.append(no_of_words/(postprocess_end_time-postprocess_start_time))
    print("\n")
    print("input words: ", ip_words, "\n")
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