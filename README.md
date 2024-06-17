# LLM Pre/Post processing

This project profiles the pre/post processing of LLM model using Torch Profiler

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
git clone -b torch_profiler https://github.com/sudhir-mcw/llm.git
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

## Run

```
cd llm
python run_inference.py <input_words>
```
_note_ : _input words max limit is 40000_

outputs are recorded in llm/exp-logs folder, which can be visualized using tensorboard

Time profiling output will be printed
