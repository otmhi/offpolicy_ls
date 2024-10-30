# Logarithmic Smoothing for Pessimistic Off-Policy Evaluation, Selection and Learning

Source code for the paper [Logarithmic Smoothing for Pessimistic Off-Policy Evaluation, Selection and Learning](https://arxiv.org/abs/2405.14335) published at NeuRIPS 2024 (Spotlight).


## Creating the environment

A virtual Python environment can be created from the requirement file holding all the needed packages to reproduce the results of the paper.

    virtualenv -p python3.9 pess_ls
    source pess_ls/bin/activate
    pip install -r requirements.txt

## Run experiments

### Run OPE/OPS experiments

The OPE/OPS experiments are built on the experiments conducted in [Confident Off-Policy Evaluation and Selection through
Self-Normalized Importance Weighting](https://arxiv.org/abs/2006.10460), and used the associated [Github package](https://github.com/google-deepmind/offpolicy_selection_eslb).

OPE and OPS experiments are defined in the ope_ops folder, move to it if you want to run them:

To run OPE experiments, run:

    python policy_evaluation.py

To run OPS experiments, run:

    python policy_selection.py

### Run OPL experiments

OPL experiments are defined in the opl folder, move to it if you want to run them:

To run OPL experiments, run:

    python policy_learning.py

## Citing this work
If you use this code, please cite our work

    @misc{sakhi2024logarithmicsmoothingpessimisticoffpolicy,
      title={Logarithmic Smoothing for Pessimistic Off-Policy Evaluation, Selection and Learning}, 
      author={Otmane Sakhi and Imad Aouali and Pierre Alquier and Nicolas Chopin},
      year={2024},
      eprint={2405.14335},
      url={https://arxiv.org/abs/2405.14335}}
