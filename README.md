# Logarithmic Smoothing for Pessimistic Off-Policy Evaluation, Selection and Learning

Source code for the paper ["Logarithmic Smoothing for Pessimistic Off-Policy Evaluation, Selection and Learning" - Otmane Sakhi, Imad Aouali, Pierre Alquier, Nicolas Chopin](https://proceedings.neurips.cc/paper_files/paper/2024/hash/9379ea6ba7a61a402c7750833848b99f-Abstract-Conference.html) published at NeuRIPS 2024 (Spotlight).


## Creating the environment

A virtual Python environment can be created from the requirement file holding all the needed packages to reproduce the results of the paper.

    virtualenv -p python3.9 pess_ls
    source pess_ls/bin/activate
    pip install -r requirements.txt

## Run experiments

### Run OPE/OPS experiments

The OPE/OPS experiments are built on the experiments conducted in ["Kuzborskij, I., Vernade, C., Gyorgy, A., & Szepesvári, C. (2021, March). Confident off-policy evaluation and selection through self-normalized importance weighting. In International Conference on Artificial Intelligence and Statistics (pp. 640-648). PMLR."](https://arxiv.org/abs/2006.10460). 

The code for these experiments is heavily inspired from their associated [Github package](https://github.com/google-deepmind/offpolicy_selection_eslb).
    
To run OPE experiments, please execute:

    python ope_ops/policy_evaluation.py

To run OPS experiments, please execute:

    python ope_ops/policy_selection.py

### Run OPL experiments

To run OPL experiments, please execute:

    python opl/policy_learning.py

## Citing this work
If you use this code, please cite our work:

    @inproceedings{NEURIPS2024_9379ea6b,
     author = {Sakhi, Otmane and Aouali, Imad and Alquier, Pierre and Chopin, Nicolas},
     booktitle = {Advances in Neural Information Processing Systems},
     editor = {A. Globerson and L. Mackey and D. Belgrave and A. Fan and U. Paquet and J. Tomczak and C. Zhang},
     pages = {80706--80755},
     publisher = {Curran Associates, Inc.},
     title = {Logarithmic Smoothing for Pessimistic Off-Policy Evaluation, Selection and Learning},
     url = {https://proceedings.neurips.cc/paper_files/paper/2024/file/9379ea6ba7a61a402c7750833848b99f-Paper-Conference.pdf},
     volume = {37},
     year = {2024}}
