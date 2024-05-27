# LLM Pre/Post processing

This project profiles the pre/post processing of LLM model using Torch Profiler

## Prerequisite
Install python 3.12

Create a virtual environment 
```
venv <path_to_python3.12> <env_name>
source <envpath>/bin/activate
```

## Installation


```bash
git clone 
cd llm
pip install -r requirements.txt
```

## Download LLAMA2 Model
Place the model just outside llm repo

```bash
cd <to_folder_outside_llm_repo>
wget
unzip
```

## Run

```
cd llm
python run_inference.py <input_words>
```
_note_ : _input words max limit is 40000_

outputs are recorded in llm/exp-logs folder, which can be visualized using tensorboard

Time profiling output will be printed