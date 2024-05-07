import torch
import transformers
import time
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import PyTorchBenchmark, PyTorchBenchmarkArguments


args = PyTorchBenchmarkArguments(models=["meta-llama/Llama-2-7b-chat-hf"], batch_sizes=[8], sequence_lengths=[8, 32])
benchmark = PyTorchBenchmark(args)

results = benchmark.run()
print(results)