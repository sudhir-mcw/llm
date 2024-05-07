import torch
import transformers
import time
from transformers import LlamaForCausalLM, LlamaTokenizer, TextStreamer
# from transformers import PyTorchBenchmark, PyTorchBenchmarkArguments


# args = PyTorchBenchmarkArguments(models=["meta-llama/Llama-2-7b-chat-hf"], batch_sizes=[8], sequence_lengths=[8, 32])
# benchmark = PyTorchBenchmark(args)

# results = benchmark.run()
# print(results)


start_time = time.time()
model_dir = "./llama-2-7b-chat-hf"

print(1)
model = LlamaForCausalLM.from_pretrained(model_dir)  #### Loading checkpoint shards:---> starts here
print(model.generation_config)
print(2)
tokenizer = LlamaTokenizer.from_pretrained(model_dir)
print(3)
prompt = "Hey, can you give me good place for my vacation ?"

inp_time = time.time()
inputs = tokenizer(prompt, return_tensors="pt")
print("inputs",inputs)
# streamer = TextStreamer(tokenizer)
print(4)
# Generate and Measure the time taken for token generation
start_tokenization_time = time.time() # ------------------
print("start_tokenization_time",start_tokenization_time)
# generate_ids,token_time_list = model.generate(inputs.input_ids, max_length=300, streamer=streamer)  #3 for streaming 
# generate_ids,token_time_list = model.generate(inputs.input_ids, max_new_tokens=10, do_sample=True,num_beams=4)
# generate_ids,token_time_list = model.generate(inputs.input_ids, max_length=300, do_sample=True,num_beams=4)  # beam search multinomial sampling
# generate_ids,token_time_list = model.generate(inputs.input_ids, max_length=300) # greedy search
generate_ids,token_time_list = model.generate(inputs.input_ids, max_length=300, do_sample=True) # multinomial sampling
# generate_ids,token_time_list = model.generate(inputs.input_ids, max_length=300, penalty_alpha=0.6, top_k=4) # contrastive search


end_tokenization_time = time.time()   # ------------------
print("generate_ids",generate_ids)

# Calculate the number of tokens
num_tokens = generate_ids.shape[-1]
print("token_time_list",token_time_list)

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