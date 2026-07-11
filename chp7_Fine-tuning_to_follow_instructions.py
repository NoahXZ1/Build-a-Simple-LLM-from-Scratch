"""This is for chpater 7: fine-tuning a model for following instructions"""
#7.2 Preparing a dataset for supervised fine-tuning instruction fine-tuning
import json
import os
import urllib

def download_and_load_file(file_path, url):
    if not os.path.exists(file_path):
        with urllib.request.urlopen(url) as response:
            text_data =response.read().decode("utf-8")
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text_data)
    with open(file_path, "r") as file:
        data = json.load(file)
    return data

file_path = "instruction_data.json"
