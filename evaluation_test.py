### This is run when you want to select the parameters from the parameters file
import transformers 
import pickle
import torch
import neptune
from knockknock import slack_sender
from transformers import *
import glob 
from transformers import BertTokenizer
from transformers import BertForSequenceClassification, AdamW, BertConfig
import random
import pandas as pd
from transformers import BertTokenizer
from Models.utils import masked_cross_entropy,fix_the_random,format_time,save_normal_model,save_bert_model
from sklearn.metrics import accuracy_score,f1_score
from tqdm import tqdm
from TensorDataset.datsetSplitter import createDatasetSplit
from TensorDataset.dataLoader import combine_features
from Preprocess.dataCollect import collect_data,set_name
from sklearn.metrics import accuracy_score,f1_score,roc_auc_score,recall_score,precision_score
import matplotlib.pyplot as plt
import time
import os
from transformers import BertTokenizer
import GPUtil
from sklearn.utils import class_weight
import json
from Models.bertModels import *
from Models.otherModels import *
import sys
import time
from waiting import wait
from sklearn.preprocessing import LabelEncoder
import numpy as np
import threading
import argparse
import ast



def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) # only difference

### gpu selection algo
def get_gpu():
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    while(1):
        tempID = [] 
        tempID = GPUtil.getAvailable(order = 'memory', limit = 1, maxLoad = 0.1, maxMemory = 0.07, includeNan=False, excludeID=[], excludeUUID=[])
        if len(tempID) > 0:
            print("Found a gpu")
            print('We will use the GPU:',tempID[0],torch.cuda.get_device_name(tempID[0]))
            deviceID=tempID
            return deviceID
        else:
            time.sleep(5)
#    return flag,deviceID


##### selects the type of model
def select_model(params,embeddings):
    if(params['bert_tokens']):
        if(params['what_bert']=='weighted'):
            model = SC_weighted_BERT.from_pretrained(
            params['path_files'], # Use the 12-layer BERT model, with an uncased vocab.
            num_labels = params['num_classes'], # The number of output labels
            output_attentions = True, # Whether the model returns attentions weights.
            output_hidden_states = False, # Whether the model returns all hidden-states.
            hidden_dropout_prob=params['dropout_bert'],
            params=params
            )
        else:
            print("Error in bert model name!!!!")
        return model
    else:
        text=params['model_name']
        if(text=="birnn"):
            model=BiRNN(params,embeddings)
        elif(text == "birnnatt"):
            model=BiAtt_RNN(params,embeddings,return_att=False,)
        elif(text == "birnnscrat"):
            model=BiAtt_RNN(params,embeddings,return_att=True)
        elif(text == "cnn_gru"):
            model=CNN_GRU(params,embeddings)
        elif(text == "lstm_bad"):
            model=LSTM_bad(params)
        else:
            print("Error in model name!!!!")
        return model


# Train shape labels - 15383
# Val shape labels - 1922
# Test shape labels - 1924
def Eval_phase(params,which_files='test',model=None,test_dataloader=None,device=None):
    if(params['is_model']==True):
        print("model previously passed")
        model.eval()
    else:
        return 1
#         ### Have to modify in the final run
#         model=select_model(params['what_bert'],params['path_files'],params['weights'])
#         model.cuda()
#         model.eval()


    print("Running eval on ",which_files,"...")
    t0 = time.time()

    # Put the model in evaluation mode--the dropout layers behave differently
    # during evaluation.
    # Tracking variables 
    
    true_labels=[]
    pred_labels=[]
    logits_all=[]
    misclassified_labels = []
    misclassified_input_ids = []
    # Evaluate data for one epoch
    for step, batch in tqdm(enumerate(test_dataloader)):

        # Progress update every 40 batches.
        if step % 40 == 0 and not step == 0:
            # Calculate elapsed time in minutes.
            elapsed = format_time(time.time() - t0)


        # `batch` contains three pytorch tensors:
        #   [0]: input ids 
        #   [1]: attention vals
        #   [2]: attention mask
        #   [3]: labels 
        b_input_ids = batch[0].to(device)
        b_att_val = batch[1].to(device)
        b_input_mask = batch[2].to(device)
        b_labels = batch[3].to(device)


        # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
        model.zero_grad()        
        outputs = model(b_input_ids,
            attention_vals=b_att_val,
            attention_mask=b_input_mask, 
            labels=None,device=device)
        logits = outputs[0]
        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        # Calculate the accuracy for this batch of test sentences.
        # Accumulate the total accuracy.
        pred_labels+=list(np.argmax(logits, axis=1).flatten())
        true_labels+=list(label_ids.flatten())
        logits_all+=list(logits)

        # getting the list of misclassification labels and input ids
        batch_preds = np.argmax(logits, axis=1).flatten()
        batch_labels = label_ids.flatten()
        input_ids = b_input_ids.to('cpu').numpy()
        misclassified_labels += list(batch_labels[np.where(batch_labels != batch_preds)])
        misclassified_input_ids += list(input_ids[np.where(batch_labels != batch_preds)])
        # print("The misclassified input ids are:", misclassified_input_ids)
    
    # saving the misclassified labels and the misclassified input ids
    # np.save("misclassified_labels.npy",misclassified_labels)
    # np.save("misclassified_input_ids.npy",misclassified_input_ids)

    logits_all_final=[]
    for logits in logits_all:
        logits_all_final.append(softmax(logits))
    
    testf1=f1_score(true_labels, pred_labels, average='macro')
    testacc=accuracy_score(true_labels,pred_labels)

    print("labels shape: ", len(true_labels),"pred_labels shape",len(pred_labels))
    


    if(params['num_classes']==3):
        testrocauc=roc_auc_score(true_labels, logits_all_final,multi_class='ovo',average='macro')
    else:
        #testrocauc=roc_auc_score(true_labels, logits_all_final,multi_class='ovo',average='macro')
        testrocauc=0
    testprecision=precision_score(true_labels, pred_labels, average='macro')
    testrecall=recall_score(true_labels, pred_labels, average='macro')
    
    if(params['logging']!='neptune' or params['is_model'] == True):
        # Report the final accuracy for this validation run.
        print(" Accuracy: {0:.4f}".format(testacc))
        print(" Fscore: {0:.4f}".format(testf1))
        print(" Precision: {0:.4f}".format(testprecision))
        print(" Recall: {0:.4f}".format(testrecall))
        print(" Roc Auc: {0:.4f}".format(testrocauc))
        print(" Test took: {:}".format(format_time(time.time() - t0)))
        # print(ConfusionMatrix(true_labels,pred_labels))
    else:
        bert_model = params['path_files']
        language  = params['language']
        name_one=bert_model+"_"+language
        neptune.create_experiment(name_one,params=params,send_hardware_metrics=False,run_monitoring_thread=False)
        neptune.append_tag(bert_model)
        neptune.append_tag(language)
        neptune.append_tag('test')
        neptune.log_metric('test_f1score',testf1)
        neptune.log_metric('test_accuracy',testacc)
        neptune.log_metric('test_precision',testprecision)
        neptune.log_metric('test_recall',testrecall)
        neptune.log_metric('test_rocauc',testrocauc)
        neptune.stop()

    return testf1,testacc,testprecision,testrecall,testrocauc,logits_all_final,pred_labels,true_labels

    
    








params_data={
    'include_special':False, 
    'bert_tokens':False,
    'type_attention':'softmax',
    'set_decay':0.1,
    'majority':2,
    'max_length':128,
    'variance':10,
    'window':4,
    'alpha':0.5,
    'p_value':0.8,
    'method':'additive',
    'decay':False,
    'normalized':False,
    'not_recollect':True,
}

#"birnn","birnnatt","birnnscrat","cnn_gru"


common_hp={
    'is_model':True,
    'logging':'local',  ###neptune /local
    'learning_rate':0.1,  ### learning rate 2e-5 for bert 0.001 for gru
    'epsilon':1e-8,
    'batch_size':16,
    'to_save':True,
    'epochs':10,
    'auto_weights':True,
    'weights':[1.0,1.0,1.0],
    'model_name':'birnnscrat',
    'random_seed':42,
    'num_classes':3,
    'att_lambda':100,
    'device':'cuda',
    'train_att':True

}
    
    
params_bert={
    'path_files':'bert-base-uncased',
    'what_bert':'weighted',
    'save_only_bert':False,
    'supervised_layer_pos':11,
    'num_supervised_heads':1,
    'dropout_bert':0.1
 }


params_other = {
        "vocab_size": 0,
        "padding_idx": 0,
        "hidden_size":64,
        "embed_size":0,
        "embeddings":None,
        "drop_fc":0.2,
        "drop_embed":0.2,
        "drop_hidden":0.1,
        "train_embed":False,
        "seq_model":"gru",
        "attention":"softmax"
}


if(params_data['bert_tokens']):
    for key in params_other:
        params_other[key]='N/A'
else:
    for key in params_bert:
        params_bert[key]='N/A'


def Merge(dict1, dict2,dict3, dict4): 
    res = {**dict1, **dict2,**dict3, **dict4} 
    return res 

params = Merge(params_data,common_hp,params_bert,params_other)


dict_data_folder={
      '2':{'data_file':'Data/dataset.json','class_label':'Data/classes_two.npy'},
      '3':{'data_file':'Data/dataset.json','class_label':'Data/classes.npy'}
}

if __name__=='__main__': 
    my_parser = argparse.ArgumentParser(description='Train a deep-learning model with the given data')

    # Add the arguments
    my_parser.add_argument('path',
                           metavar='--path_to_json',
                           type=str,
                           help='The path to json containining the parameters')
    
    my_parser.add_argument('use_from_file',
                           metavar='--use_from_file',
                           type=str,
                           help='whether use the parameters present here or directly use from file')
    
    my_parser.add_argument('attention_lambda',
                           metavar='--attention_lambda',
                           type=str,
                           help='required to assign the contribution of the atention loss')
    
    
    
    args = my_parser.parse_args()
    params['best_params']=False
    if(args.use_from_file == 'True'):
        with open(args.path,mode='r') as f:
            params = json.load(f)
        for key in params:
            if params[key] == 'True':
                 params[key]=True
            elif params[key] == 'False':
                 params[key]=False
            if( key in ['batch_size','num_classes','hidden_size','supervised_layer_pos','num_supervised_heads','random_seed','max_length']):
                if(params[key]!='N/A'):
                    params[key]=int(params[key])
                
            if((key == 'weights') and (params['auto_weights']==False)):
                params[key] = ast.literal_eval(params[key])
        params['best_params']=True 
    ##### change in logging to output the results to neptune
    params['logging']='local'
    if(params['logging']=='neptune'):
        from api_config import project_name,api_token
        neptune.init(project_name,api_token=api_token)
        neptune.set_project(project_name)
    torch.autograd.set_detect_anomaly(True)
    if torch.cuda.is_available() and params['device']=='cuda':    
        # Tell PyTorch to use the GPU.    
        device = torch.device("cuda")
        ##### You can set the device manually if you have only one gpu
        ##### comment this line if you don't want to manually set the gpu
        deviceID = get_gpu()
        torch.cuda.set_device(deviceID[0])
        ##### comment this line if you don't want to manually set the gpu
        #### parameter required is the gpu id
        #torch.cuda.set_device(0)
        
    else:
        print('Since you dont want to use GPU, using the CPU instead.')
        device = torch.device("cpu")
        
        
    #### Few handy keys that you can directly change.
    params['variance']=1
    params['epochs']=5
    params['to_save']=True
    params['num_classes']=3
    params['data_file']=dict_data_folder[str(params['num_classes'])]['data_file']
    params['class_names']=dict_data_folder[str(params['num_classes'])]['class_label']
    if(params['num_classes']==2 and (params['auto_weights']==False)):
          params['weights']=[1.0,1.0]
            
    #for att_lambda in [0.001,0.01,0.1,1,10,100]
    params['att_lambda']=float(args.attention_lambda)
    # train_model(params,device)


    embeddings=None
    if(params['bert_tokens']):
        train,val,test=createDatasetSplit(params)
        # print("Test data", len(test),test[:5])
    else:
        train,val,test,vocab_own=createDatasetSplit(params)
        params['embed_size']=vocab_own.embeddings.shape[1]
        params['vocab_size']=vocab_own.embeddings.shape[0]
        embeddings=vocab_own.embeddings
    if(params['auto_weights']):
        y_test = [ele[2] for ele in test] 
#         print(y_test)
        encoder = LabelEncoder()
        encoder.classes_ = np.load(params['class_names'],allow_pickle=True)
        params['weights']=class_weight.compute_class_weight('balanced',np.unique(y_test),y_test).astype('float32') 
        #params['weights']=np.array([len(y_test)/y_test.count(encoder.classes_[0]),len(y_test)/y_test.count(encoder.classes_[1]),len(y_test)/y_test.count(encoder.classes_[2])]).astype('float32') 
        
        
    print(params['weights'])
    train_dataloader =combine_features(train,params,is_train=True)   
    validation_dataloader=combine_features(val,params,is_train=False)
    test_dataloader=combine_features(test,params,is_train=False)
    
   
    # model=select_model(params,embeddings)
    model_path = "Saved/bert-base-uncased_11_6_3_"+str(args.attention_lambda)
    model=SC_weighted_BERT.from_pretrained(
            model_path, # Use the 12-layer BERT model, with an uncased vocab.
            num_labels = params['num_classes'], # The number of output labels
            output_attentions = True, # Whether the model returns attentions weights.
            output_hidden_states = False, # Whether the model returns all hidden-states.
            hidden_dropout_prob=params['dropout_bert'],
            params=params
            )
    if(params["device"]=='cuda'):
        model.cuda()

    test_fscore,test_accuracy,test_precision,test_recall,test_roc_auc,logits_all_final,pl,tl=Eval_phase(params,'test',model,test_dataloader,device)
    # print("ACC", np.mean(1*(pl==tl)))
    # np.save("true_labels.npy",tl)
    # np.save("predicted_labels.npy",pl)


    
