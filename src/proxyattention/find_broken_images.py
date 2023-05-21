from meta_utils import get_files
from PIL import Image
from tqdm import tqdm
from pathlib import Path
 
files = get_files(path = "/home/eragon/Documents/Datasets/places256")
files = [f for f in files if ".csv" not in str(f)]

for filename in tqdm(files):
    try:
        img = Image.open(filename) # open the image file
        img.verify() # verify that it is, in fact an image
    except (IOError, SyntaxError) as e:
        print('Bad file:', filename)
        Path(filename).unlink()