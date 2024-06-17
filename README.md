# LLM Pre/Post processing

This project run inference for LLM model and profiles for TTFT

## Prerequisite
Install python 3.10

Create a virtual environment 
```
venv <path_to_python3.10> <env_name>
source <envpath>/bin/activate
```

## Installation


```bash
git clone -b main https://github.com/sudhir-mcw/llm.git
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
python run_inference.py
```


## Run for TTFT
Replace the files from llm/util_changes with
- path_to_pyvenv/lib/python3.12/site-packages/transformers/generation/utils.py

- path_to_pyvenv/lib/python3.12/site-packages/transformers/generation/configuration_utils.py

```
cd llm
python run_inference_ttft.py
```
