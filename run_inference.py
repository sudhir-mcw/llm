import torch
import transformers
import time
from transformers import LlamaForCausalLM, LlamaTokenizer, TextStreamer
model_dir = "/home/mcwaiteam/TinyLlama-1.1B-Chat-v0.6"

model = LlamaForCausalLM.from_pretrained(model_dir)  #### Loading checkpoint shards:---> starts here
print(model.generation_config)
tokenizer = LlamaTokenizer.from_pretrained(model_dir)
prompt = "Hey, can you give me good place for my vacation ?"

inputs = tokenizer(prompt, return_tensors="pt")
print("inputs",inputs)

# Different modes of execution
# generate_ids,token_time_list = model.generate(inputs.input_ids, max_length=300, streamer=streamer)  #3 for streaming 
# generate_ids,token_time_list = model.generate(inputs.input_ids, max_new_tokens=10, do_sample=True,num_beams=4)
# generate_ids,token_time_list = model.generate(inputs.input_ids, max_length=300, do_sample=True,num_beams=4)  # beam search multinomial sampling
generate_ids,token_time_list = model.generate(inputs.input_ids, max_length=300, num_beams=4) # greedy search
# generate_ids,token_time_list = model.generate(inputs.input_ids, max_length=300, do_sample=True) # multinomial sampling
# generate_ids,token_time_list = model.generate(inputs.input_ids, max_length=300, penalty_alpha=0.6, top_k=4) # contrastive search

print("generate_ids",generate_ids)

output = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print("output",output)
