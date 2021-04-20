# master-thesis-title

[Description]


## Installation

[Installation]

### Run tests

In order to run the provided tests,
the following commands have to be executed at the base directory of the repository:

```bash
# TODO maybe not needed, just add app src dir to python path in console
# install package, to make it available to the tests
pip install -e .

# install pytest to run the tests
pip install pytest
```

After this you should be able to run the tests:

```bash
# go to tests directory
cd tests

# run tests
pytest
```


## Train Agent

[Train_Agent]


## Notebooks

### Train LSTM-DQN agent in Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pybnen/conbas/blob/master/notebooks/lstm_dqn_train.ipynb)

Use the jupyter notebook [lstm_dqn_train.ipynb](https://colab.research.google.com/github/pybnen/conbas/blob/master/notebooks/lstm_dqn_train.ipynb) to train the LSTM-DQN agent in Google Colab.

### Local jupyter notebook server

When running the notbooks locally, add root of git repro to PYTHONPATH environment variable
before starting jupyter notebook server e.g.

> export PYTHONPATH="$PYTHONPATH:/path/to/git/repo"


## Reproduce Results

[HOW_TO_REPRODUCE_RESULTS]