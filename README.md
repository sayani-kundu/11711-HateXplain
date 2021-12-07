# 11711 Assignment 3 : HateXplain: A Benchmark Dataset for Explainable Hate Speech Detection 

Contributors: Arvind Subramaniam, Aryan Mehra and Sayani Kundu

### In this repository, we replicate (and soon build upon) the results of the following paper: 

"[HateXplain: A Benchmark Dataset for Explainable Hate Speech Detection](https://arxiv.org/abs/2012.10289)". 
<!-- Please follow this [link] "" for the repo to the main paper.  -->



```
Branches
└── main - Contains the code for the main branch.
└── masked_data - Contains the code for masking the target communities in the training data.  
└── attention_modification - Contains the code for the Lenient and Conservative Attention models.
```

------------------------------------------
***Our Contributions*** 
------------------------------------------

~~~
./Experiments.ipynb                     --> Contains the codes for training, evaluation and all the metrics calculations (performance, bias and Explainability) using BERT. There are markdowns for each section in the notebook.
./evaluation_test.py                    --> Contains the code for evaluating the saved models on the test set.
./Data Analysis.ipynb                   --> Contains the code for generating the histograms and confusion matrix for the best model.
~~~

------------------------------------------
***Folder and File Description*** :open_file_folder:	
------------------------------------------
~~~

./Data                                  --> Contains the dataset related files.
./Models                                --> Contains the codes for all the classifiers used
./Preprocess  	                   --> Contains the codes for preprocessing the dataset	
./best_model_json                       --> Contains the parameter values for the best models
./Experiments.ipynb                     --> Contains the codes for all the metrics (performance, bias and Explainability) using BERT
./model_explain_output_100.json         --> The explainability output log file for lambda = 100 (attention constant)
./model_explain_output_0point001.json   --> The explainability output log file for lambda = 0.001 (attention constant)
~~~

------------------------------------------
***Saved Models*** 
------------------------------------------
Due to the large size of the saved models folder, we have shared a drive link to access the same "[here](https://drive.google.com/drive/folders/1_PVUHwvY7EHoc_w9Ebqg1dgh9R9e2p87?usp=sharing)". The folder structure is as follows:
```
Saved
└── bert-base-uncased_11_6_3_0.001
       ├── config.json
       ├── pytorch_model.bin
       ├── special_tokens_map.json  
       ├── tokenizer_config.json
       └── vocab.txt
       
       
└── bert-base-uncased_11_6_3_1
       ├── ...   
       
└── bert-base-uncased_11_6_3_10
       ├── ...     
       
└── bert-base-uncased_11_6_3_100
       ├── ...
```
------------------------------------------
***Types of Experiments*** 
------------------------------------------
We report three types of metrics for the BERT model:
```
Performance - Accuracy, F1 score and AUROC.

Bias - GMB-Subgroup-AUC, GMB-BPSN-AUC and GMB-BNSP-AUC.


Explainability
     ├── Plausibility - IOU F1, Token F1 and AUPRC.
     └── Faithfulness - Comprehensiveness and Sufficiency.
```


------------------------------------------
***Usage instructions*** 
------------------------------------------
Install the corresponding version of tensorflow (listed below) to avoid version conflicts. Next, install the required libraries using the following command (preferably inside an environent)

~~~
pip install tensorflow==2.4.0
pip install -r requirements.txt
~~~

------------------------------------------
***Training*** 
------------------------------------------
To train the model use the following command.

~~~
usage: manual_training_inference.py [-h]
                                    --path_to_json --use_from_file
                                    --attention_lambda

Train a deep-learning model with the given data

positional arguments:
  --path_to_json      The path to json containining the parameters
  --use_from_file     whether use the parameters present here or directly use
                      from file
  --attention_lambda  required to assign the contribution of the atention loss

~~~

------------------------------------------
***Bias Calculation*** 
------------------------------------------
We convert the task into a binary classification problem by grouping the 3 labels into two groups as follows:

```
1. **hate speech** and **offensive**  - Toxic
2. **Normal** - Normal
```
The following script can be used to run evaluation for bias calculation (unintended bias towards a target community)

```
python testing_for_bias.py bert_supervised 0.001
```

------------------------------------------
***Explainability calculation*** 
------------------------------------------
The Explainability aspect of the model is an indicator of how convincing the text is to a human interpretor (plausibility) and the reasoning capability (faithfulness)
The following script can be used to run evaluation for Explainability calculation (unintended bias towards a target community)

```
python testing_for_rational.py bert_supervised 0.001
```


The value of lambda (attention constant) is mentioned as the last argument. For both Bias and Explainability, we have tested for different values of lambda such as 0.001, 1, 10 and 100.
