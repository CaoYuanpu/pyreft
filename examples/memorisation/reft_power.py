import copy, json, random, re
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence
import pandas as pd
import matplotlib.pyplot as plt
from plotnine import ggplot, aes, geom_line, theme_minimal
from matplotlib.ticker import MaxNLocator
plt.rcParams.update({'font.size': 20, 'font.family': 'Sans'})
import torch
import transformers
from datasets import Dataset
from transformers import Trainer

from pyreft import (
    TaskType,
    get_reft_model,
    ReftConfig,
    ReftTrainerForCausalLM, 
    ReftDataCollator,
    ReftSupervisedDataset,
    make_last_position_supervised_data_module,
    ConsreftIntervention
)

IGNORE_INDEX = -100
prompt_no_input_template = """<s>[INST] <<SYS>>
You are a helpful assistant.
<</SYS>>

%s [/INST]
"""
device = "cuda" if torch.cuda.is_available() else "cpu"

def max_char_match_length(retrieved, golden):
    n_c, n = 0, 0
    for char in retrieved:
        if char == golden[n]:
            n_c += 1
        else:
            break
        n += 1 
    if len(retrieved) == 0:
        return 0.0
    return round(n_c/len(retrieved), 2)

make_supervised_data_module = make_last_position_supervised_data_module

# load model (take 1 min)
model_name_or_path = "meta-llama/Llama-2-7b-chat-hf" # yahma/llama-7b-hf or yahma/llama-13b-hf
model = transformers.AutoModelForCausalLM.from_pretrained(
    model_name_or_path, torch_dtype=torch.bfloat16, device_map=device)

# get tokenizer
model_max_length = 2048
tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_name_or_path, model_max_length=model_max_length, 
    padding_side="right", use_fast=False)
tokenizer.pad_token = tokenizer.unk_token

TARGET_LAYER = 15

# get reft model
reft_config = ReftConfig(representations={
    "layer": TARGET_LAYER, "component": "block_output",
    "intervention": ConsreftIntervention(
    embed_dim=model.config.hidden_size,
    low_rank_dimension=1)})
reft_model = get_reft_model(model, reft_config)
reft_model.print_trainable_parameters()

# get training data to train our intervention to remember the following sequence
memo_sequence = """
Welcome to the Natural Language Processing Group at Stanford University!
We are a passionate, inclusive group of students and faculty, postdocs
and research engineers, who work together on algorithms that allow computers
to process, generate, and understand human languages. Our interests are very
broad, including basic scientific research on computational linguistics,
machine learning, practical applications of human language technology,
and interdisciplinary work in computational social science and cognitive
science. We also develop a wide variety of educational materials
on NLP and many tools for the community to use, including the Stanza
toolkit which processes text in over 60 human languages.
"""
# data_module = make_last_position_supervised_data_module(
#     tokenizer, model, [prompt_no_input_template % "^^^&&&&&"], [memo_sequence])

data_module = make_last_position_supervised_data_module(
    tokenizer, model, ["RANDID->"], [memo_sequence])

# train
training_args = transformers.TrainingArguments(
    num_train_epochs=1000.0, output_dir="./tmp", learning_rate=2e-3, report_to=[])
trainer = ReftTrainerForCausalLM(
    model=reft_model, tokenizer=tokenizer,
    args=training_args, **data_module)
_ = trainer.train()


# prompt = tokenizer(prompt_no_input_template % "Summarize the following text: ^^^&&&&&", return_tensors="pt").to("cuda")
prompt = tokenizer("Summarize the following text: RANDID->", return_tensors="pt").to("cuda")
base_unit_location = prompt["input_ids"].shape[-1] - 1  # last position

_, reft_response = reft_model.generate(
    prompt, unit_locations={"sources->base": (None, [[[base_unit_location]]])},
    intervene_on_prompt=True, max_new_tokens=512, do_sample=False, 
    eos_token_id=tokenizer.eos_token_id, early_stopping=True
)
print(tokenizer.decode(reft_response[0], skip_special_tokens=True))

# prompt = tokenizer(prompt_no_input_template % "^^^&&&&&", return_tensors="pt").to("cuda")
prompt = tokenizer("RANDID->", return_tensors="pt").to("cuda")
base_unit_location = prompt["input_ids"].shape[-1] - 1  # last position
_, reft_response = reft_model.generate(
    prompt, unit_locations={"sources->base": (None, [[[base_unit_location]]])},
    intervene_on_prompt=True, max_new_tokens=512, do_sample=False, 
    eos_token_id=tokenizer.eos_token_id, early_stopping=True
)
print(tokenizer.decode(reft_response[0], skip_special_tokens=True))


# reft_model.set_device("cpu") # send back to cpu before saving.
# reft_model.save(
#     save_directory="./memo_demo", 
#     save_to_hf_hub=False, 
# )
