# Exploring How Fair Model Representations Relate to Fair Recommendations
This is the official implementation of the paper "Exploring How Fair Model Representations Relate to Fair Recommendations". 

## Requirements
The implementation is written in python. We have relied on multiple python packages for different aspects: data processing (numpy, pandas), metrics/simple models(scikit-learn), plotting(plotly,matplotlib), deep learning(pytorch) and prototyping/interactive coding(jupyter notebook). A comprehensive list of standard python libraries and their version is found in the "requirements.txt" file.

The code also depends on implementations of two different papers:
 - [Flexible Generation of Preference Data for Recommendation Analysis](https://arxiv.org/abs/2407.16594)
 - [Opt-in Transparent Fairness for Recommender Systems](https://link.springer.com/chapter/10.1007/978-3-031-88708-6_23)

 We borrowed an earlier version of the formers code for generating synthetic datasets. This code is found in the file "genrec_orig.py".**NOTE: We did some minor changes to the code to suit our needs, but made sure to indicate where changes have been made, see file for more details**
 
 The latter implementation was used for processing the movielens dataset, and for the implementations of all 4 considered VAE-based recommender systems. We forked their code and added the following key changes/features: early stopping for all relevant training loops (important for our varied experiments), added a function call that allows passing terminal commands as function arguments and the returnal of test-set model probabilities. *The aim is to get the fork accepted as new branch in their repository, but for the sake of the review process and reproducability, we have included the fork in the "optin" folder.*

 ## How to run
 The file "experiment_runner.py" is set up to run most of our experiments. Since most of the tests have a single core setup ran multiple times while varying a single hyperparameter, we ran our experiments by explicitly changing a set of boolean hyperparameters in the file + minor tweaking of run names/specific changes etc. Thus, we do not currently support command-line arguments or command files, but we will look into this and clean up the code a bit to make it more readable after submission.