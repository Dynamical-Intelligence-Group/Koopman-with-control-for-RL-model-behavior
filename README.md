# Koopman-with-control-for-RL-model-behavior

This repository hosts the code for "Interpreting Reinforcement Learning Model Behavior via Koopman with Control" Redman (2026). 

## Training RL models and computing the stability and controllability of their behaviors

To train RL models, open the ```KoopmanTaskRL.ipynb``` notebook. This allows you to choice an environment (e.g., CartPole, LunarLander) and training algorithm (e.g., PPO, A2C). ```n_init``` independent models are trained, the resulting models being saved. The training epochs at which these models are saved is set by ```training_epochs```. Koopman operator with control models are thne fit to the states and actions of the intermediate models, and from the approximated linear control model ```A``` and ```B```, the stability and controllability properties are computed. All the quantities are saved into the ```results``` folder, into a folder specific to that type of training algorithm.

## Comparing RL models across training 

To compare multiple differently trained RL models, open the ```KoopmanTaskRLPlotting.ipynb``` notebook. This generates plots for all specified models and training epochs. 

## Acknowledgements 

We thank Dr. Jordan Garrett for assistance with developing the code for running and analyzing the RL models. 

<img src="https://github.com/Dynamical-Intelligence-Group/Koopman-with-control-for-RL-model-behavior/blob/main/results/CartPole-v1-summary-results.png" alt="Alt Text" height = "400" width = "900">

