# Proxy Attention : Masters Thesis Project
- Using outputs of XAI algorithms as part of the training process in order to simulate the effects of Attention for a NN.
- This is a work in progress and this README will reflect changes in code or methodology.
- Implemented using Pytorch (no training frameworks used although an implementation using fast.ai was also created as a prototype.)

## Research Questions
- Is it possible to create an augmentation technique based on Attention maps?
- Is it possible to approximate the effects of Attention from ViTs in a CNN?
- Is it possible to make a network converge faster and consequently require less data using the outputs from XAI techniques?
- Does Proxy Attention impact the explainability of the model in turn?

## How to Use
- Since all the scripts have not been written yet, for now it is just possible to open pure_pytorch_src/main.ipynb 
- src/main_runner.py can also be looked at although the implementation in EXTREMELY slow.

## Directory Structure
### Main folder
- pure_pytorch_src is the main folder. It will be renamed later on.
- main.ipynb has all the runner code that is being refactored to make it a script. It will eventually just have a demo.
- the runs folder has tensorboard logs. This will not be pushed to github for now to account for storage constraints.
- meta_utils.py has utility functions that are used multiple times and refactored in this file.
- config.py will store configurations with their names for easier experiment running.

### Test
- src is a test folder that will be deleted after clean up. It was made as a quick and dirty prototype of the idea for the thesis
- This has a working implementation of the entire loop but using fast.ai
- Since it worked but proved to be too slow to use, the entire codebase will be re-written from scratch without fast.ai

### Other branches
- A julia branch exists with most of the code also written in julia + flux
- This was abandoned as it proved to be too much to implement from scratch


## Implementation Progress
- [x] Data loader
- [x] Captum integration
- [x] Stratified kfold
- [x] Transfer learning
- [x] Basic config
- [x] Data augmentation
- [x] Implemented proxy attention using outputs from Captum
- [x] Training loop with the following optimizations
	- [x] Gradient scaling
	- [x] Mixed precision training
	- [x] Progress bars 
	- [x] Tensorboard logging (mostly)
	- [] Better logging and model saving
	- [] Checkpoints
	- [] Number of epochs in between loops (Epochs -> Proxy -> Epochs ...)
	- [] More optimizations

- [] Full proxy attention loop as a function
- [] Support for config.py
- [] Support for more XAI algorithms
- [] Batch runner script
