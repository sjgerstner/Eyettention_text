import numpy as np
import os
from utils import *
import pandas
from sklearn.model_selection import StratifiedKFold, KFold
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from transformers import BertTokenizerFast
from model import Eyettention
from sklearn.preprocessing import LabelEncoder
from torch.nn.functional import cross_entropy, softmax
from collections import deque
import pickle
import argparse

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='run Eyettention on SB-SAT dataset')
	parser.add_argument(
		'--test_mode',
		help='New Sentence Split: text, New Reader Split: subject',
		type=str,
		default='text'
	)
	parser.add_argument(
		'--atten_type',
		help='attention type: global, local, local-g',
		type=str,
		default='local-g'
	)
	parser.add_argument(
		'--save_data_folder',
		help='folder path for saving results',
		type=str,
		default='./results/SB-SAT/'#CHANGED
	)
	parser.add_argument(
		'--scanpath_gen_flag',
		help='whether to generate scanpath',
		type=int,
		default=1
	)
	parser.add_argument(
		'--max_pred_len',
		help='if scanpath_gen_flag is True, you can determine the longest scanpath that you want to generate, which should depend on the sentence length',
		type=int,
		default=2048#CHANGED
	)
	parser.add_argument(
		'--gpu',
		help='gpu index',
		type=int,
		default=0
	)
	args = parser.parse_args()
	gpu = args.gpu

	#use FastTokenizer lead to warning -> The current process just got forked
	os.environ["TOKENIZERS_PARALLELISM"] = "false"
	torch.set_default_tensor_type('torch.FloatTensor')
	availbl = torch.cuda.is_available()
	print(torch.cuda.is_available())
	if availbl:
		device = f'cuda:{gpu}'
		torch.cuda.set_device(gpu)
	else:
		device = 'cpu'
		#torch.cuda.set_device(cpu)
	print(device)

	cf = {"model_pretrained": "bert-base-cased",
			"lr": 1e-3,
			"max_grad_norm": 10,
			"n_epochs": 1000,
			"n_folds": 5,
			"dataset": 'SB-SAT',
			"atten_type": args.atten_type,
			"batch_size": 2,#CHANGED TODO
			#following 4 lines CHANGED. Can't get more than 512 tokens into BERT!#TODO
			"max_sn_len": 148, #max number of words in a sentence, include start token and end token, #or 512? #needs to be the actual sn len + 2 for the CLS
			"max_sn_token": 183, #maximum number of tokens a sentence includes. include start token and end token
			"max_sp_len": 396, #max number of words in a scanpath, include start token and end token #or 1130? #needs to be the actual sp len + 2 for the CLS
			"max_sp_token": 512, #maximum number of tokens a scanpath includes. include start token and end token UNNECESSARY
			"norm_type": 'z-score',
			"earlystop_patience": 20,
			"max_pred_len":args.max_pred_len,
			"device":device
			}

	#Encode the label into interger categories, setting the exclusive category 'cf["max_sn_len"]-1' as the end sign
	le = LabelEncoder()
	le.fit(np.append(np.arange(-cf["max_sn_len"]+3, cf["max_sn_len"]-1), cf["max_sn_len"]-1))
	#le.classes_

	#load corpus
	word_info_df, sn_list, eyemovement_df = load_corpus(cf["dataset"])

	#only use native speaker
	#Make list with reader index
	reader_list = eyemovement_df.Session_Name_.unique() #CHANGED

	#Split training&test sets by text or reader, depending on configuration
	if args.test_mode == 'text':
		print('Start evaluating on new sentences.')
		split_list = sn_list
	elif args.test_mode == 'subject':
		print('Start evaluating on new readers.')
		split_list = reader_list

	#CHANGED adapt number of folds
	n_folds = min(cf["n_folds"], len(split_list))#TODO does it work?. Does work with 2 in second place
	kf = KFold(n_splits=n_folds, shuffle=True, random_state=0)
	fold_indx = 0
	#ADDED
	best_llh = -np.inf
	#for scanpath generation
	sp_dnn_list = []
	sp_human_list = []
	for train_idx, test_idx in kf.split(split_list):
		loss_dict = {'val_loss':[], 'train_loss':[], 'test_ll':[], 'test_AUC':[]}
		list_train = [split_list[i] for i in train_idx]
		list_test = [split_list[i] for i in test_idx]

		# create train validation split for training the models:
		kf_val = KFold(n_splits=n_folds, shuffle=True, random_state=0)
		for train_index, val_index in kf_val.split(list_train):
			# we only evaluate a single fold
			break
		list_train_net = [list_train[i] for i in train_index]
		list_val_net = [list_train[i] for i in val_index]

		if args.test_mode == 'text':
			sn_list_train = list_train_net
			sn_list_val = list_val_net
			sn_list_test = list_test
			reader_list_train, reader_list_val, reader_list_test = reader_list, reader_list, reader_list

		elif args.test_mode == 'subject':
			reader_list_train = list_train_net
			reader_list_val = list_val_net
			reader_list_test = list_test
			sn_list_train, sn_list_val, sn_list_test = sn_list, sn_list, sn_list

		#initialize tokenizer
		tokenizer = BertTokenizerFast.from_pretrained(cf['model_pretrained'])
		#Preparing batch data
		dataset_train = satdataset(word_info_df, eyemovement_df, cf, reader_list_train, sn_list_train, tokenizer)
		train_dataloaderr = DataLoader(dataset_train, batch_size = cf["batch_size"], shuffle = True, drop_last=True)

		dataset_val = satdataset(word_info_df, eyemovement_df, cf, reader_list_val, sn_list_val, tokenizer)
		val_dataloaderr = DataLoader(dataset_val, batch_size = cf["batch_size"], shuffle = False, drop_last=True)

		dataset_test = satdataset(word_info_df, eyemovement_df, cf, reader_list_test, sn_list_test, tokenizer)
		test_dataloaderr = DataLoader(dataset_test, batch_size = cf["batch_size"], shuffle = False, drop_last=False)

		#z-score normalization for gaze features
		fix_dur_mean, fix_dur_std = calculate_mean_std(dataloader=train_dataloaderr, feat_key="sp_fix_dur", padding_value=0, scale=1000)
		landing_pos_mean, landing_pos_std = calculate_mean_std(dataloader=train_dataloaderr, feat_key="sp_landing_pos", padding_value=0)
		sn_word_len_mean, sn_word_len_std = calculate_mean_std(dataloader=train_dataloaderr, feat_key="sn_word_len")

		# load model
		dnn = Eyettention(cf)

		#training
		episode = 0
		optimizer = Adam(dnn.parameters(), lr=cf["lr"])
		dnn.train()
		dnn.to(device)
		av_score = deque(maxlen=100)
		old_score = 1e10
		save_ep_couter = 0
		print('Start training')
		for episode_i in range(episode, cf["n_epochs"]+1):
			dnn.train()
			print('episode:', episode_i)
			counter = 0
			for batchh in train_dataloaderr:
				print(batchh)
				break
			break
		break