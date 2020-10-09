# SQLova Experiments

## 1. Description
This folder contains implementation of **interactive SQLova parser**, which uses SQLova as a base semantic parser in our MISP framework:
- Please follow [2. General Environment Setup](#2.-General-Environment-Setup) and set up the environment/data;
- For testing interactive SQLova on the fly (our EMNLP'19 setting), see [3. MISP with SQLova (EMNLP'19)](#3.-MISP-with-SQLova-(EMNLP'19));
- For learning SQLova from user interaction (our EMNLP'20 setting), see [4. Learning SQLova from user interaction (EMNLP'20)](#4.-Learning-SQLova-from-user-interaction-(EMNLP'20)).

The implementation is adapted from [the SQLova repository](https://github.com/naver/sqlova). 
Please cite the following papers if you use the code:

```
@inproceedings{yao2020imitation,
  title={An Imitation Game for Learning Semantic Parsers from User Interaction},
  author={Yao, Ziyu and Tang, Yiqi and Yih, Wen-tau and Sun, Huan and Su, Yu},
  booktitle={Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  year={2020}
}

@inproceedings{yao2019model,
  title={Model-based Interactive Semantic Parsing: A Unified Framework and A Text-to-SQL Case Study},
  author={Yao, Ziyu and Su, Yu and Sun, Huan and Yih, Wen-tau},
  booktitle={Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)},
  pages={5450--5461},
  year={2019}
}

@article{hwang2019comprehensive,
  title={A Comprehensive Exploration on WikiSQL with Table-Aware Word Contextualization},
  author={Hwang, Wonseok and Yim, Jinyeung and Park, Seunghyun and Seo, Minjoon},
  journal={arXiv preprint arXiv:1902.01069},
  year={2019}
}
```

## 2. General Environment Setup
### Environment
- Please install the Anaconda environment from [`gpu-py3.yml`](../gpu-py3.yml):
    ```
    conda env create -f gpu-py3.yml
    ```

- Download Pretrained BERT model from [here](https://drive.google.com/file/d/1f_LEWVgrtZLRuoiExJa5fNzTS8-WcAX9/view?usp=sharing) as
 `SQLova_model/download/bert/pytorch_model_uncased_L-12_H-768_A-12.bin`.

### Data
We have the pre-processed [WikiSQL data](https://github.com/salesforce/WikiSQL) available: [data.tar](https://www.dropbox.com/s/younbzsa0t6wzan/data.tar?dl=0).
Please download and uncompress it via `tar -xvf data.tar` as a folder `SQLova_model/download/data`.

If you would like to preprocess the WikiSQL data (or your own data) from scratch, please follow the [`data_preprocess.sh`](../scripts/sqlova/data_preprocess.sh) script.


## 3. MISP with SQLova (EMNLP'19)
### 3.1 Model training
To train SQLova on the full training set, please revise `SETTING=full_train` in [scripts/sqlova/pretrain.sh](../scripts/sqlova/pretrain.sh).
In the main directory, run:
```
bash scripts/sqlova/pretrain.sh
```

### 3.2 Model testing without interaction
To test SQLova regularly, in [scripts/sqlova/test.sh](../scripts/sqlova/test.sh), please revise `SETTING` 
to ensure that the model checkpoint is loaded from the desired `MODEL_DIR` folder and revise `TEST_JOB` for testing on WikiSQL dev/test set.
In the main directory, run:
``` 
bash scripts/sqlova/test.sh
```

### 3.3 Model testing with simulated user interaction
To test SQLova with human interaction under the MISP framework, in [scripts/sqlova/test_with_interaction.sh](../scripts/sqlova/test_with_interaction.sh),
revise `SETTING` to ensure that the model checkpoint is loaded from the desired `MODEL_DIR` folder and revise `DATA` for testing on WikiSQL dev/test set.
In the main directory, run:
```
bash scripts/sqlova/test_with_interaction.sh
```


## 4. Learning SQLova from user interaction (EMNLP'20)
Throughout the experiments, we consider three initialization settings:
- `SETTING=online_pretrain_1p` for using 1% of full training data for initialization;
- `SETTING=online_pretrain_5p` for using 5% of full training data for initialization;
- `SETTING=online_pretrain_10p` for using 10% of full training data for initialization.

Please revise the `SETTING` variable in each script accordingly.

### 4.1 Pretraining

#### 4.1.1 Pretrain by yourself
Before interactive learning, we pretrain the SQLova parser with a small subset of the full training data. 
Please revise `SETTING` in [scripts/sqlova/pretrain.sh](../scripts/sqlova/pretrain.sh) accordingly for different initialization settings.
Then in the main directory, run:
```
bash scripts/sqlova/pretrain.sh
```

#### 4.1.2 Use our pretrained checkpoints
You can also use our pretrained checkpoints: [initialization_checkpoints_folder.tar](https://www.dropbox.com/s/rcmz56h0803sz8g/initialization_checkpoints_folder.tar?dl=0). 
Please download and uncompress the folder via `tar -xvf initialization_checkpoints_folder.tar` and place the content as:
```
|- SQLova_model
|   |-- checkpoints_onlint_pretrain_1p
|       |-- model_best.pt
|       |-- model_bert_best.pt
|   |-- checkpoints_onlint_pretrain_5p
|   |-- checkpoints_onlint_pretrain_10p
```

#### 4.1.3 Test the pretrained models
To test the pretrained parser without user interaction, see [3.2 Model testing without interaction](#3.2-Model-testing-without-interaction).
To test the pretrained parser with simulated user interaction, see [3.3 Model testing with simulated user interaction](#3.3-Model-testing-with-simulated-user-interaction).
Make sure the `SETTING` variable is set correctly.

### 4.2 Interactive learning

The training script for each algorithm can be found below. Please run them in the main directory and 
remember to set `SETTING` accordingly for different initialization settings.

| Algorithm  | Script |
| ------------- | ------------- |
| MISP_NEIL  | [`scripts/sqlova/misp_neil.sh`](../scripts/sqlova/misp_neil.sh)  |
| Full Expert  | [`scripts/sqlova/full_expert.sh`](../scripts/sqlova/full_expert.sh)  |
| Binary User  | [`scripts/sqlova/bin_user.sh`](../scripts/sqlova/bin_user.sh)  |
| Binary User+Expert  | [`scripts/sqlova/bin_user_expert.sh`](../scripts/sqlova/bin_user_expert.sh)  |
| Self Train  | [`scripts/sqlova/self_train_0.5.sh`](../scripts/sqlova/self_train_0.5.sh)  |
| MISP_NEIL*  | [`scripts/sqlova/misp_neil_perfect.sh`](../scripts/sqlova/misp_neil_perfect.sh)  |

