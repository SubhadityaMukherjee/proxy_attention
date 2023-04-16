"""
Run result aggregator
- Read all tensorbord logs and save as a pandas dataframe for analysis
"""

import os
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
import pickle
import argparse

parser = argparse.ArgumentParser(description="Process some training options.")

parser.add_argument(
    "-i", "--save_images", action="store_true", help="Save images to table"
)
args = parser.parse_args()

# main_path = "/mnt/d/CODE/thesis_runs/proper_runs"

computer_choice = "pc"
# pc, cluster

# Make dirs
if computer_choice == "linux":
    main_path = (
        "/run/media/eragon/HDD/CODE/Github/improving_robotics_datasets/src/runs/"
    )
    main_ds_dir = "/run/media/eragon/HDD/Datasets/"

elif computer_choice == "pc":
    main_path = (
        "/mnt/d/CODE/thesis_runs/proper_runs/"
    )
    main_ds_dir = "/mnt/d/Datasets/"

os.environ["TORCH_HOME"] = main_ds_dir



def save_pickle(*args, fname="pickler.pkl") -> None:
    with open(fname, "wb") as f:
        pickle.dump(args, f)


def read_pickle(fname="pickler.pkl"):
    with open(fname, "rb") as f:
        obj = pickle.load(f)
    return obj


def get_event_files(main_path) -> list:
    """Return a list of event files under the given directory"""
    all_files = []
    for root, _, filenames in os.walk(main_path):
        for filename in filenames:
            if "events.out.tfevents" in filename:
                all_files.append(str(Path(root) / Path(filename)))
    return all_files


def process_event_acc(event_acc, save_ims=False) -> dict:
    """Process the EventAccumulator and return a dictionary of tag values"""
    all_tags = event_acc.Tags() # Get all tags
    temp_dict = {} # Store all values here
    for tag in all_tags.keys(): # Loop over all tags
        if tag == "scalars":
            # Process scalars
            for subtag in all_tags[tag]:
                try:
                    # Try to get the last value
                    temp_dict[subtag] = [
                        tag[-1] for tag in event_acc.Scalars(tag=subtag)
                    ][-1].value
                except:
                    # If there is only one value, get that
                    temp_dict[subtag] = [tag for tag in event_acc.Scalars(tag=subtag)][
                        -1
                    ].value
        if tag == "tensors":
            # Process tensors
            for subtag in all_tags["tensors"]:
                tensor_proto = event_acc.Tensors(subtag)[0].tensor_proto
                if "/text_summary" in subtag:
                    # Decode text summaries to ascii and remove the subtag suffix
                    subtag = subtag.replace("/text_summary", "")
                    value = tensor_proto.string_val[0].decode("ascii")
                else:
                    # Decode other tensors to float
                    value = tensor_proto
                temp_dict[subtag] = value

        if save_ims and tag == "images":
            # Process images
            for subtag in all_tags["images"]:
                try:
                    # Try to get the last value
                    encoded_image = event_acc.Images(subtag)[1].encoded_image_string
                except IndexError:
                    # If there is only one value, get that
                    encoded_image = event_acc.Images(subtag).encoded_image_string

                # Decode the image and save it to the dictionary
                image = Image.open(BytesIO(encoded_image))
                temp_dict[subtag] = image

    return temp_dict


def process_runs(main_path, save_ims=False) -> pd.DataFrame:
    """Process all runs and return a dataframe of all results"""
    all_files = get_event_files(main_path=main_path)
    all_dict = {}
    for files in tqdm(all_files, total=len(all_files)):
        try:
            # Process each file using the EventAccumulator and save to a dictionary
            event_acc = EventAccumulator(files)
            event_acc.Reload()
            temp_dict = process_event_acc(event_acc, save_ims=save_ims)
            all_dict[files] = temp_dict
        except IndexError:
            pass
    return pd.DataFrame.from_records(all_dict).T.reset_index()


if args.save_images: # Save images to table
    combined_df = process_runs(main_path=main_path, save_ims=True)
else: # Don't save images to table
    combined_df = process_runs(main_path=main_path, save_ims=False)

# Save the dataframe to a pickled csv
save_pickle(combined_df, fname="./results/aggregated_runs.csv")
