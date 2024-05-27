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
# from transformers import PyTorchBenchmark, PyTorchBenchmarkArguments

# from memory_profiler import profile

# args = PyTorchBenchmarkArguments(models=["meta-llama/Llama-2-7b-chat-hf"], batch_sizes=[8], sequence_lengths=[8, 32])
# benchmark = PyTorchBenchmark(args)

# results = benchmark.run()
# print(results)


PROFILE = True

'''
    profiling function uses torch.prolfer.profile(),
    logs to a file name with functions name  
    usage : 
    @profile
    def fun_to_profile():
        pass
'''
def profile(func):
    def profiler(*args, **kwargs):
        if not PROFILE:
            # print("skipped profiling")
            return func(*args, **kwargs)
        # print(psutil.cpu_percent(percpu=True,interval=0.5))
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU],
            # schedule=torch.profiler.schedule(wait=1, warmup=1, active=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(f'./exp-logs/{func.__name__}'),
            # on_trace_ready=torch.profiler.tensorboard_trace_handler(f'./exp-logs'),
            record_shapes=True,
            profile_memory=True,
            with_stack=True
            # with_flops=True,
            # with_modules=True
            ) as prof:
            generated_ids   =    func(*args, **kwargs)
        # print(prof.key_averages().table(),"completed profiling ",func.__name__)
        # print(psutil.cpu_percent(percpu=True,interval=0.5))

        return generated_ids
 
    return profiler

def load_model(model_dir=None):
    if not model_dir:
        model_dir  = "/home/mcwaiteam/TinyLlama-1.1B-Chat-v0.6"
    model = LlamaForCausalLM.from_pretrained(model_dir)  #### Loading checkpoint shards:---> starts here
    # print(model.generation_config)
    return model
def load_tokenizer(model_dir=None):
    if not model_dir:
        model_dir  = "/home/mcwaiteam/TinyLlama-1.1B-Chat-v0.6"
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
    output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return output


if __name__ == "__main__":
    shutil.rmtree(r"./exp-logs/preprocess")
    shutil.rmtree(r"./exp-logs/postprocess")
    model = load_model()
    tokenizer = load_tokenizer()
    # print("loaded model and tokenizer")
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
    ip_words = 5000

    for i in range(0, 5):
        prompt = prompt_str.join(words_list[:ip_words])
        tokenization_start_time = time.time()
        tokenized_inputs = preprocess(tokenizer, prompt)
        tokenization_end_time = time.time()
        pre_time.append(((tokenization_end_time-tokenization_start_time)*1000)/ip_words)
        pre_total.append(((tokenization_end_time-tokenization_start_time)*1000))
        no_tokens = tokenized_inputs["input_ids"].shape[1]
        gen_token.append(no_tokens)
        # print(type(print("type of tokenized_inputs ids:", type(tokenized_inputs))))
        # print("no of tokens", tokenized_inputs["input_ids"].shape[1])
        # print("tokenized_inputs ids:", tokenized_inputs)
        WT_ratio.append(ip_words/no_tokens)
        TPS.append(no_tokens/(tokenization_end_time-tokenization_start_time))

        # infernence_start_time = time.time()
        # generated_ids, timings = inference(model,tokenized_inputs)
        # infernence_end_time = time.time()
        # print("generated prc type: ", type(generated_ids))
        # print("generated ids:", generated_ids)
        # # print("decoding completed generated ids")

        postprocess_start_time = time.time()
        postprocessed_output = postprocess(tokenizer,tokenized_inputs['input_ids'])

        # print("post prc type: ", type(postprocessed_output))
        # print("post prc : ", (postprocessed_output))
        postprocess_end_time = time.time()
        op_list = postprocessed_output.split(" ")
        no_of_words = len(op_list)
        pro_time.append(((postprocess_end_time-postprocess_start_time)*1000)/no_tokens)
        pro_total.append((postprocess_end_time-postprocess_start_time)*1000)
        
        TW_ratio.append(no_tokens/no_of_words)
        WPS.append(no_of_words/(postprocess_end_time-postprocess_start_time))
    print("\n")
    print("input words: ", ip_words, "\n")
    print("Pretime: ", mean(pre_time))
    print("Total pretime ; ", mean(pre_total))
    print("TPS: ", mean(TPS))
    print("W:T ratio: ", mean(WT_ratio))
    print("\n")
    print("ip token to post proc: ", mean(gen_token))
    print("Protime: ", mean(pro_time))
    print("Total protime: ", mean(pro_total))
    print("WPS: ", mean(WPS))
    print("T:W ratio: ", mean(TW_ratio))
    print("\n")
        # print("preprocess time : ",tokenization_end_time-tokenization_start_time)
        # # # print("inference  time : ",infernence_end_time-infernence_start_time) 
        # print("postprocess  time : ",postprocess_end_time-postprocess_start_time) 

    # num_tokens = generated_ids[0].shape[-1]
    # # infer_time = infernence_end_time - infernence_start_time
    # # tokens_per_second = infer_time/num_tokens
    # print("Number of tokens:", num_tokens)
    # # print("Time taken for tokenization (seconds):", tokenization_time)
    # # print("Tokens per second:", tokens_per_second)
    # print("Total timigs :", timings)
    # print("no of tokens",num_tokens)
    

# start_time = time.time()
#model_dir = "/DATA1/LLM_repo/steps/llama/llama-2-7b-chat-hf"
# model_dir  = "/home/mcwaiteam/TinyLlama-1.1B-Chat-v0.6"
# print(1)
# model = LlamaForCausalLM.from_pretrained(model_dir)  #### Loading checkpoint shards:---> starts here
# print(model.generation_config)
# print(2)
# tokenizer = LlamaTokenizer.from_pretrained(model_dir)
# print(3)


# prompt = "Hey, can you give me good place for my vacation ?"

# inp_time = time.time()
# with torch.profiler.profile(
#     activities=[torch.profiler.ProfilerActivity.CPU],
#     # schedule=torch.profiler.schedule(wait=1, warmup=1, active=1),
#     on_trace_ready=torch.profiler.tensorboard_trace_handler('./logs/preprocess'),
#     record_shapes=True,
#     profile_memory=True,
#     with_stack=True,
#     # with_flops=True,
#     # with_modules=True
#     ) as prof:
#     inputs = tokenizer(prompt, return_tensors="pt")
# print(prof.key_averages().table(),"completed profiling")
# print("inputs",inputs)
# streamer = TextStreamer(tokenizer)
# print(4)
# Generate and Measure the time taken for token generation
# start_tokenization_time = time.time() # ------------------
# print("start_tokenization_time",start_tokenization_time)
# generate_ids,token_time_list = model.generate(inputs.input_ids, max_length=300, streamer=streamer)  #3 for streaming 
# generate_ids,token_time_list = model.generate(inputs.input_ids, max_new_tokens=10, do_sample=True,num_beams=4)
#generate_ids,token_time_list = model.generate(inputs.input_ids, max_length=50, num_beams=4)  # beam search multinomial sampling
# with torch.profiler.profile(
#     activities=[torch.profiler.ProfilerActivity.CPU],
#     # schedule=torch.profiler.schedule(wait=1, warmup=1, active=1),
#     on_trace_ready=torch.profiler.tensorboard_trace_handler('./logs/inference_postprocess'),
#     record_shapes=True,
#     profile_memory=True,
#     with_stack=True,
#     # with_flops=True,
#     # with_modules=True
#     ) as profiler:

#     generate_ids = model.generate(inputs.input_ids, max_length=50, num_beams=4)
# print(profiler.key_averages().table(),"completed profiling")

# generate_ids,token_time_list = model.generate(inputs.input_ids, max_length=300) # greedy search
# generate_ids,token_time_list = model.generate(inputs.input_ids, max_length=300, do_sample=True) # multinomial sampling
# generate_ids,token_time_list = model.generate(inputs.input_ids, max_length=300, penalty_alpha=0.6, top_k=4) # contrastive search

# generate_ids = model.generate(inputs.input_ids, max_length=50, num_beams=4)

# end_tokenization_time = time.time()   # ------------------
# print("generate_ids",generate_ids)
'''
# Calculate the number of tokens
num_tokens = generate_ids[0].shape[-1]
#print("token_time_list",token_time_list)

# Calculate the time taken for tokenization
tokenization_time = end_tokenization_time - start_tokenization_time

# Calculate tokens per second
tokens_per_second = num_tokens / tokenization_time
print(5)
output = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print("output",output)
end_time = time.time()

print("Number of tokens:", num_tokens)
print("Time taken for tokenization (seconds):", tokenization_time)
print("Tokens per second:", tokens_per_second)
print("Total inference time :", end_time - start_time)

# time_for_each_token_list = []
# for j in range(len(token_time_list)):
#     if j>0:
#         time_for_each_token_list.append(token_time_list[j] - token_time_list[j-1])
#     else:
#         time_for_each_token_list.append(token_time_list[j] - start_tokenization_time)
# print("Time taken for processing output for each token = ",time_for_each_token_list)
'''
