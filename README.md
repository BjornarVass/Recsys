# Recsys: Specialization project
The code for my specialization project

# Requirements
Python 3  
PyTorch, with CUDA support  
Pickle  
Numpy  

Tensorboard requirements(optional):
Tensorflow  
Torchvision  
Scipy  

# Data

## Datasets

LastFM:http://www.dtic.upf.edu/~ocelma/MusicRecommendationDataset/lastfm-1K.html  
Reddit:https://www.kaggle.com/colemaclean/subreddit-interactions

All models expect the preprocessed data to be found in predefined paths e.g. datasets/lastfm/, but these can easily be switched out. 


## Preprocessing
There are two different files concerned with the preprocessing: preprocess_timestamp.py and preprocess_trimmed.py

preprocess_timestamp.py is the regular and general preprocessing scheme used when the data contains global timestamps from different timezones. This is used for the two larger datasets which are to be used in the model-setup found in temporal_user.py as well as in the baseline found in inter.py

preprocess_trimmed.py is specifically tuned for the LastFM dataset, and extracts users from countries observing the CET timezone and also reduces the number of artists. This results in a smaller dataset where the fact that all timestamps belongs to the same timezone can be used. A model that does this can be found in temporal_daytime.py.


# Running of code

## Running
For simple runs that are bound to the terminal window, simply call: python [filename]  
If both python 2.x and python 3.x is available in your environment, you might have to replace "python" with "python3"

An easy way to decouple the running script from the terminal window and write to a logfile, can be achieved by: python -u [filename] &> [logfile] &  
The "-u" flag is used to make the scripts write to the file in "real-time", "&>" pipes stdout and stderr(for potential debugging) and the last "&" decouples the terminal  
The logfile in question will contain the full result tables for each epoch as well as some progress prints.

## Files
inter.py: Recommendation baseline  
inter_context.py: Recommendation baseline extended with time-gap and user context  
temporal_user.py: Proposed model without use of day-time embedding  
temporal_daytime.py: Proposed model with use of day-time embedding  

## inter.py and inter_context.py
Set the dataset by setting the "dataset" variable, and redefine paths if necessary. Hyper-parameters are all defined in the top of the file and can be tuned for different experiments. The session-representation scheme is selected by setting the "use_hidden" variable to True if the use of the last hidden state as session representations is wanted, or False if the average embedding pooling representation is wanted.

## temporal_ files
Same as for the baseline when it comes to datasets and hyper-parameters and session-representation scheme. These setups have quite a few more hyper parameters that can be tweaked.

Training schedule: The different experimental setups are set by changing the "timeless" variable which makes sure the time loss is not trained on when it is set to True. The regular training schedule is achieved by setting this to False, and this is the default option. 
