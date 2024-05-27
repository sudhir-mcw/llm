import torch
import transformers
import time
from transformers import LlamaForCausalLM, LlamaTokenizer, TextStreamer
import psutil

import os
import sys

# from umlaut import Benchmark, BenchmarkSupervisor, MemoryMetric, CPUMetric,TimeMetric

# bm = Benchmark('sample_db_file.db', description="Database for the Github sample measurements")

# bloat_metrics = {
#     # "memory": MemoryMetric('bloat memory', interval=0.1),
#     "time" :TimeMetric('time'),
#     "cpu": CPUMetric('preprocess cpu', interval=0.1)
# }


# from transformers import PyTorchBenchmark, PyTorchBenchmarkArguments

# from memory_profiler import profile

# args = PyTorchBenchmarkArguments(models=["meta-llama/Llama-2-7b-chat-hf"], batch_sizes=[8], sequence_lengths=[8, 32])
# benchmark = PyTorchBenchmark(args)

# results = benchmark.run()
# print(results)


PROFILE = True

# '''
#     profiling function uses torch.prolfer.profile(),
#     logs to a file name with functions name  
#     usage : 
#     @profile
#     def fun_to_profile():
#         pass
# '''
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

# @BenchmarkSupervisor(bloat_metrics.values(), bm)
@profile
def preprocess(tokenizer,prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    return inputs


def inference(model,tokenized_inputs):
    generate_ids = model.generate(tokenized_inputs.input_ids, max_length=50, num_beams=4)
    return generate_ids

# @profile
def postprocess(tokenizer,generated_ids):
    output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return output


if __name__ == "__main__":

    model = load_model()
    tokenizer = load_tokenizer()
    # print("loaded model and tokenizer")

    # prompt = "Give me some places to visit on vacation?"

    prompt = "lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum"

    tokenization_start_time = time.time()
    tokenized_inputs = preprocess(tokenizer,prompt)
    tokenization_end_time = time.time()
    # bm.close()

    # infernence_start_time = time.time()
    # generated_ids = inference(model,tokenized_inputs)
    # infernence_end_time = time.time()

    # # print("decoding completed generated ids")

    # postprocess_start_time = time.time()
    # postprocessed_output = postprocess(tokenizer,generated_ids)
    # postprocess_end_time = time.time()

    # print("preprocess time : ",tokenization_end_time-tokenization_start_time)
    # print("inference  time : ",infernence_end_time-infernence_start_time) 
    # print("postprocess  time : ",postprocess_end_time-postprocess_start_time) 

    # num_tokens = generated_ids[0].shape[-1]
    # print("no of tokens",num_tokens)
    
