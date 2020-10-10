# EditSQL Experiments

## 1. Description
This folder contains implementation of **interactive EditSQL parser**, which uses EditSQL as a base semantic parser in our MISP framework:
- Please follow [2. General Environment Setup](#2-general-environment-setup) and set up the environment/data;
- For testing interactive EditSQL on the fly (our EMNLP'19 setting), see [3. MISP with EditSQL](#3-misp-with-editsql);
- For learning EditSQL from user interaction (our EMNLP'20 setting), see [4. Learning EditSQL from user interaction (EMNLP'20)](#4-learning-editsql-from-user-interaction-emnlp20).

The implementation is adapted from [the EditSQL repository](https://github.com/ryanzhumich/editsql). 
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

@InProceedings{zhang2019editing,
  author =      "Rui Zhang, Tao Yu, He Yang Er, Sungrok Shim, Eric Xue, Xi Victoria Lin, Tianze Shi, Caiming Xiong, Richard Socher, Dragomir Radev",
  title =       "Editing-Based SQL Query Generation for Cross-Domain Context-Dependent Questions",
  booktitle =   "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
  year =        "2019",
  address =     "Hong Kong, China"
}
```

## 2. General Environment Setup
### Environment
- Please install the Anaconda environment from [`gpu-py3.yml`](../gpu-py3.yml):
    ```
    conda env create -f gpu-py3.yml
    ```
- Download the Glove word embedding from [here](https://nlp.stanford.edu/projects/glove/) and put it as `EditSQL/word_emb/glove.840B.300d.txt`.

- Download Pretrained BERT model from [here](https://drive.google.com/file/d/1f_LEWVgrtZLRuoiExJa5fNzTS8-WcAX9/view?usp=sharing) as `EditSQL/model/bert/data/annotated_wikisql_and_PyTorch_bert_param/pytorch_model_uncased_L-12_H-768_A-12.bin`


### Data
We have the pre-processed and cleaned [Spider data](https://yale-lily.github.io/spider) available: [data_clean.tar](https://www.dropbox.com/s/tmj4qnzemxvi5bo/data_clean.tar?dl=0).
Please download and uncompress it via `tar -xvf data_clean.tar` as a folder `EditSQL/data_clean`. 
Note that the training set has been cleaned with its size reduced (see [our paper](https://arxiv.org/pdf/2005.00689.pdf), Appendix B.3 for details).


## 3. MISP with EditSQL
We explain how to build and test EditSQL under MISP following our EMNLP'19 setting.

### 3.1 Model training
To train EditSQL on the full training set, please revise `SETTING=''` (empty string) in [scripts/editsql/pretrain.sh](../scripts/editsql/pretrain.sh).
In the main directory, run:
```
bash scripts/editsql/pretrain.sh
```

### 3.2 Model testing without interaction
To test EditSQL (trained on the full training set) regularly, in [scripts/editsql/test.sh](../scripts/editsql/test.sh), 
please revise `SETTING=''` (empty string) to ensure the `LOGDIR` loads the desired model checkpoint.
In the main directory, run:
``` 
bash scripts/editsql/test.sh
```

### 3.3 Model testing with simulated user interaction
To test EditSQL (trained on the full training set) with human interaction under the MISP framework, in [scripts/editsql/test_with_interaction.sh](../scripts/editsql/test_with_interaction.sh),
revise `SETTING='full_train'` to ensure the `LOGDIR` loads the desired model checkpoint.
In the main directory, run:
```
bash scripts/editsql/test_with_interaction.sh
```


## 4. Learning EditSQL from user interaction (EMNLP'20)
### 4.1 Pretraining

#### 4.1.1 Pretrain by yourself
Before interactive learning, we pretrain the EditSQL parser with 10% of the full training set. 
Please ensure `SETTING='_10p'` in [scripts/editsql/pretrain.sh](../scripts/editsql/pretrain.sh).
Then in the main directory, run:
```
bash scripts/editsql/pretrain.sh
```
When the training is finished, please rename and move the best model checkpoint from `EditSQL/logs_clean/logs_spider_editsql_10p/pretraining/save_X` 
to `EditSQL/logs_clean/logs_spider_editsql_10p/model_best.pt`.

#### 4.1.2 Use our pretrained checkpoint
You can also use our pretrained checkpoint: [logs_clean.tar](https://www.dropbox.com/s/4n6dg0xcru91smu/logs_clean.tar?dl=0).
Please download and uncompress the content as `EditSQL/logs_clean/ogs_spider_editsql_10p/model_best.pt`.


#### 4.1.3 Test the pretrained model
To test the pretrained parser without user interaction, see [3.2 Model testing without interaction](#32-model-testing-without-interaction).
To test the pretrained parser with simulated user interaction, see [3.3 Model testing with simulated user interaction](#33-model-testing-with-simulated-user-interaction).
Make sure `SETTING=online_pretrain_10p` is set in the scripts.

### 4.2 Interactive learning

The training script for each algorithm can be found below. Please run them in the main directory.

| Algorithm  | Script |
| ------------- | ------------- |
| MISP_NEIL  | [`scripts/editsql/misp_neil.sh`](../scripts/editsql/misp_neil.sh)  |
| Full Expert  | [`scripts/editsql/full_expert.sh`](../scripts/editsql/full_expert.sh)  |
| Self Train  | [`scripts/editsql/self_train_0.5.sh`](../scripts/editsql/self_train_0.5.sh)  |
| MISP_NEIL*  | [`scripts/editsql/misp_neil_perfect.sh`](../scripts/editsql/misp_neil_perfect.sh)  |



