import torch
import transformers

from transformers import LlamaForCausalLM, LlamaTokenizer

model_dir = "./llama-2-7b-chat-hf"
model = LlamaForCausalLM.from_pretrained(model_dir)

tokenizer = LlamaTokenizer.from_pretrained(model_dir)

print("Creating pipeline start ------")
pipeline = transformers.pipeline(
"text-generation",
model=model,
tokenizer=tokenizer,
torch_dtype=torch.float16,
device_map="cpu",
)
print("Creating pipeline end ------")

print("Actual pipeline start")
sequences = pipeline(
'I just have a ball. teach me how to play cricket ?\n',
do_sample=True,
top_k=10,
num_return_sequences=1,
eos_token_id=tokenizer.eos_token_id,
max_length=700,
)
print("Actual pipeline end")
for seq in sequences:
    print(f"{seq['generated_text']}")