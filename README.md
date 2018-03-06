# Hieararchical RNN recommender with temporal modeling
The code for my master's thesis

# Requirements
Python 3  
PyTorch, with CUDA support  
Numpy  
Scipy  

# Data

## Datasets

LastFM:http://www.dtic.upf.edu/~ocelma/MusicRecommendationDataset/lastfm-1K.html  
Reddit:https://www.kaggle.com/colemaclean/subreddit-interactions

All models expect the preprocessed data to be found in predefined paths e.g. datasets/lastfm/. The paths can/has to be changed if another file structure is to be used.


## Preprocessing
There are two different files concerned with the preprocessing: preprocess_general.py and preprocess_trimmed.py

preprocess_general.py is the regular and general preprocessing scheme used when the data contains global timestamps from different timezones. This is used for the two larger datasets which are to be used in the hierarchical, contextual and temporal setup of the model. This preprocesser can be set to either split the data by timestamp or a fractional split of each users sessions. This is done through the bool SPLIT_ON_TIMESTAMP. A secondary option of setting a maxmum standard deviation of inter-session time-gaps (user-wise) , can also be set. This only affects the generation of an extra version of the dataset, where users who violates this threshold are removed. 

preprocess_trimmed.py is specifically tuned for the LastFM dataset, and extracts users from countries observing the CET timezone and also reduces the number of artists. This results in a smaller dataset where the fact that all timestamps belongs to the same timezone can be used.  


# Running of code

## Running
For simple runs that are bound to the terminal window, simply call: python [filename]  
If both python 2.x and python 3.x is available in your environment, you might have to replace "python" with "python3"

As of now, only the main script/model, dynamic_model.py, will automatically write to a new log file. For the hawkes baseline and non-hierarchical recommender basline, the logs are directly printed to stdout.

An easy way to decouple a running script from the terminal window and pipe stdout to a logfile, can be achieved by:  
- python -u [filename] &> [logfile] &  
The "-u" flag is used to make the scripts write to the file in "real-time", "&>" pipes stdout and stderr(for potential debugging) and the last "&" decouples the terminal  
The logfile in question will contain the full result tables for each epoch as well as some progress prints.

## Files
intra.py: Non-hierarchical recommendation baseline  
dynamic_model.py: Dynamic model that can be set to hieararchical baseline, contextual and temporal model  
hawkes_baseline.py: Temporal prediction baseline  

## dynamic_model.py
The model is specified near the top of the file under the comment "#runtime settings".
- dataset: determines which dataset to use  
- flags["context"]: is set to True if contexts are to be used (contextual and temporal setup)  
- flags["temporal"]: is set to True if temporal setup is to be used  
- SEED: sets the random seed (for logging purposes)  
- GPU: spcify which GPU to use in case there are more than one  
- directory: specify which directory logs will be outputted in  
- debug: set to True if debugging

Note that if both flags are set to False, the model setup is that of the hierarchical baseline.