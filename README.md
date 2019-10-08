# Model-based Interactive Semantic Parsing (MISP)

This repository provides code implementations for the EMNLP'19 paper "[Model-based Interactive Semantic Parsing:
A Unified Framework and A Text-to-SQL Case Study](http://web.cse.ohio-state.edu/~yao.470/paper/MISP_EMNLP19.pdf)".

**IMPORTANT: The code implementation in this branch (`emnlp`), which adopts a binary Q&A interaction, is released to reproduce the performance reported in our paper. A cleaner, better refactored version, which adopts a multi-choice Q&A interaction, can be found in branch `multichoice_q` and is more recommended.**

## 1. Introduction
As a promising paradigm, _interactive semantic parsing_ has shown to improve both semantic parsing accuracy and user confidence in the results. To facilitate its research, we propose **Model-based Interactive Semantic Parsing (MISP)**, a unified framework that views the interactive semantic parsing problem as designing a _model-based_ intelligent agent. The following figures show an overview of MISP and its instantiation (**MISP-SQL**) for text-to-SQL parsing.

<p align="center">
<img src="https://github.com/sunlab-osu/MISP/blob/multichoice_q/MISP.png" alt="MISP framework" title="MISP framework" width="350" border="10"/> <img src="https://github.com/sunlab-osu/MISP/blob/multichoice_q/text2sql.png" alt="A case study of MISP on text-to-SQL parsing" title="A case study of MISP on text-to-SQL parsing" width="400" border="10"/>
</p>

A MISP agent maintains an **agent state** and has three major components:
* **World Model**, which perceives the environment signals and predicts the future based on the agent's internal knowledge.
* **Error Detector**, which introspects its states and decides whether and where human intervention is needed.
* **Actuator**, which realizes the agent's action in a user-friendly way, e.g., by generating a natural language question.

This repository contains the implementation of MISP-SQL when the base semantic parser is SQLNet [(Xu et al., 2017)](https://arxiv.org/pdf/1711.04436.pdf), SQLova [(Hwang et al., 2019)](https://arxiv.org/pdf/1902.01069.pdf) or SyntaxSQLNet [(Yu et al., 2018)](https://arxiv.org/pdf/1810.05237.pdf). However, the framework potentially can also be applied to other base semantic parsers. We provide two versions of implementations:
* Branch `emnlp`: The _original_ version supporting binary-choice Q&A interaction, which can be used to reproduce our EMNLP results. To use this version, please switch the branch by `git checkout emnlp`.
* Branch `multichoice_q` (__default, recommended__): A _refactored_ version supporting multi-choice Q&A interaction. This is the default branch of this repository.

Please cite our work if you use our implementation:
```
@InProceedings{yao2019model,
  author =      "Ziyu Yao, Yu Su, Huan Sun, Wen-tau Yih",
  title =       "Model-based Interactive Semantic Parsing: A Unified Framework and A Text-to-SQL Case Study",
  booktitle =   "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
  year =        "2019",
  address =     "Hong Kong, China"
}
```

## 2. Experiments

Our experiments involve three different base semantic parsers: SQLNet, SQLova and SyntaxSQLNet. This section shows how to run MISP with each of the base semantic parsers.

### 2.1 MISP with SQLNet
#### Requirements
* Python 2.7
* Pytorch 0.2.0
* Please follow [SQLNet's instruction](https://github.com/xiaojunxu/SQLNet#installation) to install other dependencies.

Note: It is recommended to create a conda environment ([download](https://www.anaconda.com/distribution/), [usage](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)) to install the above packages. If you do so, you may put your environment name [here](interaction_SQLNet.sh#L3), i.e., `source activate ENV_NAME`.

#### Data preparation
Please download the `.tar` file from [here](https://www.dropbox.com/s/rjaz5t3mnj84tpk/sources.tar?dl=0) and uncompress it under the folder `SQLNet_model`. 

#### To run the interactive system
Under the main directory (please revise the output paths accordingly before running the script):
```
bash interaction_SQLNet.sh
```

### 2.2 MISP with SQLova
#### Requirements
* Python 3.6
* Pytorch 1.0.1
* Please follow [SQLova's instruction](https://github.com/naver/sqlova#requirements) to install other dependencies.
Note: If you have created a new conda environment to install the above packages, please put its name [here](interaction_sqlova.sh#L3), i.e., `source activate ENV_NAME`.

#### Data preparation
Please download the `.tar` file from [here](https://www.dropbox.com/s/p1q59bpzjyk0h5e/sources.tar?dl=0) and uncompress it under the folder `SQLova_model`. 

#### To run the interactive system
Under the main directory (please revise the output paths accordingly before running the script):
```
bash interaction_sqlova.sh
```

### 2.3 MISP with SyntaxSQLNet
#### Requirements
* Python 2.7
* Pytorch 0.2.0

(You can use the same environment as SQLNet for running SyntaxSQLNet).

#### Data preparation
Please download the `.tar` file from [here](https://www.dropbox.com/s/vi65ezm8k1j92zd/sources.tar?dl=0) and uncompress it under the folder `syntaxSQL`. Please also download and unzip the pretrained [Glove](https://nlp.stanford.edu/data/wordvecs/glove.42B.300d.zip) word embedding, and put it as `syntaxSQL/glove/glove.42B.300d.txt`. 

#### To run the interactive system
Under the main directory (please revise the output paths accordingly before running the script):
```
bash interaction_syntaxSQL.sh
```

## 3. Acknowledgement
The implementations of MISP-SQL applied to SQLNet, SQLova and SyntaxSQLNet are adapted from their non-interactive implementations: 
* [SQLNet](https://github.com/xiaojunxu/SQLNet)
* [SQLova](https://github.com/naver/sqlova)
* [SyntaxSQLNet](https://github.com/taoyds/syntaxSQL)


