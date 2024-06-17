# LLM Pre/Post processing

This project profiles the pre/post processing of LLM model by python multi-process threads

## Machice Requirements
- Arch: x86_64 or arm64
- OS  : Ubuntu 18 to 22
- RAM : Min 6Gb
- ROM : Min 64Gb

## Prerequisite
Install python 3.10

Create a virtual environment 
```
venv <path_to_python3.10> <env_name>
source <envpath>/bin/activate
```

## Installation


```bash
git clone -b multi_core https://github.com/sudhir-mcw/llm.git
cd llm
pip install -r requirements.txt
```

## Download LLAMA2 Model
Place the model just outside llm repo

```bash
cd <to_folder_outside_llm_repo>
wget https://drive.google.com/file/d/1ny1GhyIvT-hPoad3ROewgqQ5YsrGWZ8f/view?usp=drive_link
unzip TinyLlama-1.1B-Chat-v0.6.zip
```

## Run torch profiler for 4 cores

```
cd llm
python run_inference_torch.py <input_words>
```
_note_ : _input words max limit is 40000_

outputs are recorded in llm/exp-logs folder, which can be visualized using tensorboard

Time profiling output will be printed

## Run Default timeit profiler for 4 cores

```
cd llm
python run_inference_timeit.py <input_words>
```
_note_ : _input words max limit is 40000_

Time profiling output will be printed

## Run Memray Memory Profiler for 4 cores

```
cd llm
python3 -m memray run -o output.bin run_inference.py <input_words>
python3 -m memray flamegraph output.bin
```
positional arguments:
  {run,flamegraph,table,live,tree,parse,summary,stats}

                        Mode of operation
    run                 Run the specified application and track memory usage
    flamegraph          Generate an HTML flame graph for peak memory usage
    table               Generate an HTML table with all records in the peak memory usage
    live                Remotely monitor allocations in a text-based interface
    tree                Generate a tree view in the terminal for peak memory usage
    parse               Debug a results file by parsing and printing each record in it
    summary             Generate a terminal-based summary report of the functions that allocate most memory
    stats               Generate high level stats of the memory usage in the terminal

Above options give different reports for visualization

## Run CPU Profiler for 4 cores
_note_ : _Output is not accurate, because it gives overall cpu time not specific to seperate cores_
```
cd llm
python run_inference_cpu_cycles.py <input_words>
```

_note_ : _FIL memory profiler does not support multi processing_