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
import pysnooper

import argparse

parser = argparse.ArgumentParser(description="Process some training options.")

parser.add_argument(
    "-i", "--save_images", action="store_true", help="Save images to table"
)
# parser.add_argument('-c', '--continue_training', action='store_true', help='Continue the training from where it left off.')

args = parser.parse_args()


main_path = "./runs/"


def get_event_files(main_path, save_ims=False):
    """Return a list of event files under the given directory"""
    all_files = []
    for root, _, filenames in os.walk(main_path):
        for filename in filenames:
            if "events.out.tfevents" in filename:
                all_files.append(str(Path(root) / Path(filename)))
    return all_files


# @pysnooper.snoop()
def process_event_acc(event_acc, save_ims=False):
    """Process the EventAccumulator and return a dictionary of tag values"""
    all_tags = event_acc.Tags()
    temp_dict = {}
    for tag in all_tags.keys():
        if tag == "scalars":
            for subtag in all_tags[tag]:
                # print(subtag)
                try:
                    temp_dict[subtag] = [
                        tag[-1] for tag in event_acc.Scalars(tag=subtag)
                    ][-1].value
                except:
                    temp_dict[subtag] = [tag for tag in event_acc.Scalars(tag=subtag)][
                        -1
                    ].value
         # Process tensors
        # if "tensors" in all_tags:

        if tag == "tensors":
            for subtag in all_tags["tensors"]:
                tensor_proto = event_acc.Tensors(subtag)[0].tensor_proto
                if "/text_summary" in subtag:
                    subtag = subtag.replace("/text_summary", "")
                    value = tensor_proto.string_val[0].decode("ascii")
                else:
                    value = tensor_proto
                temp_dict[subtag] = value

        # Process images
        if save_ims and tag == "images":
            for subtag in all_tags["images"]:
                try:
                    encoded_image = event_acc.Images(subtag)[1].encoded_image_string
                except IndexError:
                    encoded_image = event_acc.Images(subtag).encoded_image_string
                image = Image.open(BytesIO(encoded_image))
                temp_dict[subtag] = image
       
    return temp_dict


def process_runs(main_path, save_ims=False):
    all_files = get_event_files(main_path=main_path, save_ims=False)
    all_dict = {}
    for files in tqdm(all_files, total=len(all_files)):
        # print(files)
        try:
            event_acc = EventAccumulator(files)
            event_acc.Reload()
            temp_dict = process_event_acc(event_acc, save_ims=save_ims)
            all_dict[files] = temp_dict
        except IndexError:
            pass
    return pd.DataFrame.from_records(all_dict).T.reset_index()


if args.save_images:
    combined_df = process_runs(main_path=main_path, save_ims=True)
else:
    combined_df = process_runs(main_path=main_path, save_ims=False)
save_pickle(combined_df, fname="./results/aggregated_runs.csv")
