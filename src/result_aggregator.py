"""
Run result aggregator
- Read all tensorbord logs and save as a pandas dataframe for analysis
"""

import os
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tqdm import tqdm
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import clipboard
import base64
from io import BytesIO
from PIL import Image
from proxyattention.meta_utils import save_pickle

main_path = "./runs/"


def get_event_files(main_path):
    """Return a list of event files under the given directory"""
    all_files = []
    for root, _, filenames in os.walk(main_path):
        for filename in filenames:
            if "events.out.tfevents" in filename:
                all_files.append(str(Path(root) / Path(filename)))
    return all_files


def process_event_acc(event_acc):
    """Process the EventAccumulator and return a dictionary of tag values"""
    all_tags = event_acc.Tags()
    temp_dict = {}
    for tag in all_tags.keys():
        if tag == "scalars":
            for subtag in all_tags[tag]:
                temp_dict[subtag] = [tag[-1] for tag in event_acc.Scalars(tag=subtag)][
                    -1
                ]
        if tag == "tensors":
            for subtag in all_tags[tag]:
                temp_dict[subtag.replace("/text_summary", "")] = (
                    [tag[-1] for tag in event_acc.Tensors(tag=subtag)][0]
                    .string_val[0]
                    .decode("ascii")
                )
        if tag == "images":
            for subtag in all_tags[tag]:
                temp_dict[subtag] = Image.open(
                    BytesIO(event_acc.Images(subtag)[1].encoded_image_string)
                )
    return temp_dict


def process_runs(main_path):
    all_files = get_event_files(main_path=main_path)
    all_dict = {}
    for files in tqdm(all_files, total=len(all_files)):
        event_acc = EventAccumulator(files)
        event_acc.Reload()
        temp_dict = process_event_acc(event_acc)
        all_dict[files] = temp_dict
    return pd.DataFrame.from_records(all_dict).T.reset_index()


combined_df = process_runs(main_path=main_path)
save_pickle(combined_df, fname = "./results/aggregated_runs.csv")
