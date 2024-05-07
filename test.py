import sys

# # Be sure we're using the user's site-packages instead of root's
# INTRINSIC_SITE_PKGS = '/home/intrinsic/.local/lib/python3.8/site-packages'
# if INTRINSIC_SITE_PKGS not in sys.path:
#   print("using intrinsic site-packages")
#   sys.path.insert(0, INTRINSIC_SITE_PKGS)

import llama

import os
import time
import torch

ROOT_DIR = "/DATA1/LLM_repo/steps/llama"
CKPT_DIR = os.path.join(ROOT_DIR, "llama-2-7b-chat")
TOKENIZER_PATH = os.path.join(ROOT_DIR, "tokenizer.model")
MAX_SEQ_LEN = 2048

print("loading model...")
model = llama.Llama.build(
    ckpt_dir = CKPT_DIR,
    tokenizer_path = TOKENIZER_PATH,
    max_seq_len = MAX_SEQ_LEN,
    max_batch_size = 1,
)
print("model loaded.")

print("Begin warmup executions...")
for i in range(3):
	start = time.time()
	completion = model.text_completion(prompts=["What is the weather in New York?"], max_gen_len=100)
	duration = time.time() - start
	print(f"warmup exec'ed in {duration:.2f}s")

print("executing 4real")
# start = time.time()
# torch.cuda.cudart().cudaProfilerStart()
# completion = model.text_completion(prompts=["What is the weather in Washington DC?"], max_gen_len=100)
# torch.cuda.cudart().cudaProfilerStop()
# duration = time.time() - start
# print(f"inference exec'ed in {duration:.2f}s")
# print(completion[0]["generation"])