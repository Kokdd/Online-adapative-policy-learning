# Online Adapative Policy making

**Author:** Ruiduo Jia

This is a git repository to release the code based on the algorithms of [Online Multi-Armed Bandits with Adaptive Inference (neurips2021)](https://proceedings.neurips.cc/paper_files/paper/2021/hash/0ec04cb3912c4f08874dd03716f80df1-Abstract.html) 

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. 

### Installing

First clone the git repository

```
git@github.com:Kokdd/Online-adapative-policy-learning.git
```

The code is designed for python 3.10 (although we believe any version of python 3.X would work with it). We recommend using Anaconda Python at first. We mostly reply on three basic packages(numpy/scipy/pandas) and we also use matplotlib and openpyxl to generate frames and figures. 

The following code would run an experiment of the algorithms.

```
cd Online-adapative-policy-learning
python Run.py
```



## Code structure

- **Action.py** All the arms are defined under the class "Action".
- **Observation.py** Given the arm, we would observe a outcome based on the arm's reward and this "observe" movement is defined under the class Observation.
- **Function_prop_score.py** This includes the function used to calculate the propensity score of the arms.
- **Run.py** The main body of the code, whose design strictly follows the algorithms in the paper. There are parameters that people can change at the last loop, including the SNR, the times of the initial observations, etc.

## Authors

* **Ruiduo Jia**



