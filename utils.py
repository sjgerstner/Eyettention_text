#coding=utf-8
import numpy as np
import pandas as pd
import os
from typing import Tuple
import re
from torch.utils.data import Dataset
import torch
from tqdm import tqdm

def load_bsc() -> Tuple[pd.DataFrame, ...]:
	"""
	:return: word info dataframe, part-of-speech info, eye movements
	"""
	bsc_path = './Data/beijing-sentence-corpus/'
	info_path = os.path.join(bsc_path, 'BSC.Word.Info.v2.xlsx')
	bsc_emd_path = os.path.join(bsc_path, 'BSC.EMD/BSC.EMD.txt')
	word_info_df = pd.read_excel(info_path, 'word')
	pos_info_df = pd.read_excel(info_path, header=None)
	eyemovement_df = pd.read_csv(bsc_emd_path, delimiter='\t')
	return word_info_df, pos_info_df, eyemovement_df

def load_corpus(corpus, by_page=True):
	if corpus == 'BSC':
		#load word data, POS data, EM data
		word_info_df, pos_info_df, eyemovement_df = load_bsc()
		return word_info_df, pos_info_df, eyemovement_df
	elif corpus == 'celer':
		eyemovement_df = pd.read_csv('./Data/celer/data_v2.0/sent_fix.tsv', delimiter='\t')
		eyemovement_df['CURRENT_FIX_INTEREST_AREA_LABEL'] = eyemovement_df.CURRENT_FIX_INTEREST_AREA_LABEL.replace('\t(.*)', '', regex=True)
		word_info_df = pd.read_csv('./Data/celer/data_v2.0/sent_ia.tsv', delimiter='\t')
		word_info_df['IA_LABEL'] = word_info_df.IA_LABEL.replace('\t(.*)', '', regex=True)
		return word_info_df, None, eyemovement_df
	elif corpus == 'meco':
		import pyreadr
		# Reading the R dataset
		eye_tracking_path = '../meco/joint_fix_trimmed.rda'
		texts_path = '../meco/supp texts.csv'
		meco_rda = pyreadr.read_r(eye_tracking_path)
		# Extracting the DataFrame
		meco_df = meco_rda["joint.fix"]
		# Creating a list of sentences 5 x 14,
		sentences_df = pd.read_csv(texts_path)
		# English only
		meco_df = meco_df[meco_df['lang']=='en']
		sentences_df = sentences_df[sentences_df['Unnamed: 0']=='English']
		# Renaming the text columns to numeric indices
		text_columns = sentences_df.columns[1:-2]  # Excluding the first language column and last two unnamed columns
		renamed_columns = {name: str(index) for index, name in enumerate(text_columns, start=1)}
		sentences_df.rename(columns=renamed_columns, inplace=True)
		sentences_df.rename(columns={'Unnamed: 0': 'Language'}, inplace=True)
		sentences_df.loc[1,'Language'] = 'en'
		# Dropping the last two unnamed columns
		sentences_df.drop(columns=['Unnamed: 13', 'Unnamed: 14'], inplace=True)
		# Final 13 x 13 df
		sentences_df = sentences_df.dropna(how='all')
		# Creating a list of unique readers (subjects)
		readers = list(meco_df['uniform_id'].dropna().unique())
		# introduce newlines
		for col in sentences_df.columns:
			text = sentences_df[col].iloc[0]
			text = text.replace("\\n", "\n")
			prev_line = 1
			new_text = re.split(r'-|\s+', text)
			text_df = meco_df[meco_df['renamed_trial']==' '.join(new_text[:5])]
			nwords = len(new_text)
			for id in range(nwords):
				new_line_series = text_df.loc[text_df["wordnum"]==id+1, ["line", "line.word"]]
				if new_line_series.empty:
					continue
				new_line = int(new_line_series["line"].iloc[0])
				# print('line number', new_line)
				# print('position on line:', new_line_series["line.word"].iloc[0])
				if new_line > prev_line:
					prev_line=new_line
					words_to_go_back = int(new_line_series["line.word"].iloc[0])
					new_text[id-words_to_go_back] = new_text[id-words_to_go_back]+'\n'
			new_str = ' '.join(new_text)
			new_str = new_str.replace('\n ', '\n')
			sentences_df[col].iloc[0] = new_str
		return meco_df, sentences_df, readers
	elif corpus == 'SB-SAT':
		eyemovement_df = pd.read_csv('../SB-SAT/fixation/18sat_fixfinal.csv')
		#filter out unwanted fixations
		eyemovement_df = eyemovement_df[eyemovement_df["type"]=="reading"]
		eyemovement_df = eyemovement_df[eyemovement_df["CURRENT_FIX_INTEREST_AREA_LABEL"]!=""]
		eyemovement_df = eyemovement_df[eyemovement_df["CURRENT_FIX_INTEREST_AREA_LABEL"]!="(FIN)"]
		#the first three ids are buttons and not part of the text
		eyemovement_df = eyemovement_df[eyemovement_df.CURRENT_FIX_INTEREST_AREA_ID>3]
		eyemovement_df.CURRENT_FIX_INTEREST_AREA_ID -=3 #first word in text has id 1, then [CLS] will have 0
		#eyemovement_df['CURRENT_FIX_INTEREST_AREA_LABEL'] = eyemovement_df.CURRENT_FIX_INTEREST_AREA_LABEL.replace('\t(.*)', '', regex=True)
		#Make array with page titles #CHANGED
		if by_page:
			sn_list = list(eyemovement_df.page_name.unique())
		else:
			sn_list = list(eyemovement_df.book_name.unique())
		#text dict, entries are title:text
		#in parallel, count words per page and update ids in eyemovement_df
		text_dict = dict([(title, '') for title in sn_list])
		if by_page:
			for title in sn_list:
				with open(f'../SB-SAT/stimuli/reading screenshot/{title}.txt') as f:
					text_dict[title] = f.read()[:-1]#get rid of the text editor's newline at the end
		else:
			for title in sn_list:
				word_count=0
				for k in range(1,7):#each text has at most 6 pages
					eyemovement_df.loc[eyemovement_df["page_name"]==f'reading-{title}-{k}', "CURRENT_FIX_INTEREST_AREA_ID"] += word_count
					try:
						with open(f'../SB-SAT/stimuli/reading screenshot/reading-{title}-{k}.txt') as f:
							for line in f:
								if line=='\n':
									text_dict[title] += '\n\n'#paragraphs are marked this way
								else:
									text_dict[title] += line[:-1] + ' '#without newline as there is no new paragraph, but with a space instead
						word_count = len(text_dict[title].split())
					except:
						break
				text_dict[title] = text_dict[title][:-1]#remove the last space
		return text_dict, sn_list, eyemovement_df
	
class satdataset(Dataset):
	def __init__(self, word_info_df, eyemovement_df, cf, reader_list, sn_list, tokenizer, by_page=True):
		self.data = _process_sat(word_info_df, eyemovement_df, cf, reader_list, sn_list, tokenizer, by_page)
	def __len__(self):
		return len(self.data["SN_input_ids"])
	def __getitem__(self,idx):
		sample = {}
		sample["sn_input_ids"] = self.data["SN_input_ids"][idx,:]
		sample["sn_attention_mask"] = self.data["SN_attention_mask"][idx,:]
		sample["sn_word_len"] = self.data['SN_WORD_len'][idx,:]
		sample['word_ids_sn'] =  self.data['WORD_ids_sn'][idx,:]

		sample["sp_input_ids"] = self.data["SP_input_ids"][idx,:]
		sample["sp_attention_mask"] = self.data["SP_attention_mask"][idx,:]
		sample['word_ids_sp'] =  self.data['WORD_ids_sp'][idx,:]

		sample["sp_pos"] = self.data["SP_ordinal_pos"][idx,:]
		sample["sp_fix_dur"] = self.data["SP_fix_dur"][idx,:]
		sample["sp_landing_pos"] = self.data["SP_landing_pos"][idx,:]

		sample["sub_id"] = self.data["sub_id"][idx]

		#NEW: word indices BEFORE which a newline occurs
		sample["sn_newlines"] = self.data["SN_newlines"][idx,:]

		return sample

def _process_sat(text_dict, eyemovement_df, cf, reader_list, sn_list, tokenizer, by_page=True):
	SN_input_ids, SN_attention_mask, SN_WORD_len, WORD_ids_sn = [], [], [], []
	SP_input_ids, SP_attention_mask, WORD_ids_sp = [], [], []
	SP_ordinal_pos, SP_landing_pos, SP_fix_dur = [], [], []
	sub_id_list =  []
	SN_newlines = [] #NEW
	for sn_id in tqdm(sn_list):
		#process sentence sequence
		if by_page:
			sn_df = eyemovement_df[eyemovement_df.page_name==sn_id]
		else:
			sn_df = eyemovement_df[eyemovement_df.book_name==sn_id]
		sn = text_dict[sn_id]
		#NEW: clean final paragraph break if necessary
		if sn.endswith('\n'):
			sn = sn[:-1]
		#print(sn)
		#tokenization and padding
		tokenizer.padding_side = 'right'
		sn_str = '[CLS]' + ' ' + (sn.replace('\n\n', ' [SEP]\n')) + ' ' + '[SEP]'#CHANGED: replace double newlines with [SEP]
		#print(sn_str)
		#NEW
		line_list = sn_str.splitlines()
		#print(line_list)
		line_len = np.array([len(line.split()) for line in line_list])
		#print(line_len)
		cum_line_len = np.cumsum(line_len[:-1])#linebreak not relevant after last line
		#print(cum_line_len)
		sn_newlines = np.full(cf["max_sn_len"], False)
		for i in cum_line_len:
			sn_newlines[i] = True
		#print(sn_newlines)

		#pre-tokenized input
		tokens = tokenizer(sn_str.split(),
							add_special_tokens = False,
							truncation=True,
							max_length = cf['max_sn_token'],
							padding = 'max_length',
							return_attention_mask=True,
							is_split_into_words=True,
							return_overflowing_tokens = not by_page,
							stride=32,
							)
		#print(tokens)
		encoded_sn = tokens['input_ids']
		mask_sn = tokens['attention_mask']
		#use offset mapping to determine if two tokens are in the same word.
		#index start from 0, CLS -> 0 and SEP -> last index
		word_ids_sn = tokens.word_ids()
		#print(word_ids_sn)
		word_ids_sn = [val if val is not None else np.nan for val in word_ids_sn]
		#print(word_ids_sn)
		if by_page:
			first = 1
			last = next(i for i in reversed(range(len(word_ids_sn))) if word_ids_sn[i] is not np.nan) -1
		else:#TODO loop over the sequences in word_ids_sn?
			index = None #TODO
			#pretend we had a sequence on its own:
			#make sure the text begins with [CLS] and ends with [SEP], as it will be a training example on its own
			#no lost info thanks to the stride and having added [CLS]/[SEP] at the very beginning/end
			first = word_ids_sn[1]
			last_index = next(i for i in reversed(range(len(word_ids_sn))) if word_ids_sn[i] is not np.nan)
			last = word_ids_sn[last_index-1]
			word_ids_sn = [id - first+1 for id in word_ids_sn]
			word_ids_sn[0] = 0
			word_ids_sn[last_index] = last+1
			encoded_sn[index][0] = tokenizer.cls_token_id
			encoded_sn[index][last_index] = tokenizer.sep_token_id
		#choose corresponding fixations
		short_sn_df = sn_df[sn_df["CURRENT_FIX_INTEREST_AREA_ID"]>=first]
		short_sn_df = short_sn_df[short_sn_df["CURRENT_FIX_INTEREST_AREA_ID"]<=last]
		short_sn_df.CURRENT_FIX_INTEREST_AREA_ID -= (first-1)
		#print(short_sn_df.CURRENT_FIX_INTEREST_AREA_ID.max())
		#compute 1/word length for each word
		sn_word_len = compute_word_length_celer(np.array([len(word) for word in sn_str.split()[first:last+1]]))
		#process fixation sequence
		for sub_id in reader_list:
			sub_df = short_sn_df[short_sn_df['Session_Name_']==sub_id]
			# remove fixations on non-words
			if len(sub_df) == 0:
				#no scanpath data found for the subject
				continue

			#sub_id convertible to integer
			sub_id = sub_id[3:]

			#prepare decoder input and output
			#I don't know how to find out fixation location within the word with this dataset, hence None
			sp_word_pos, sp_fix_loc, sp_fix_dur = sub_df.CURRENT_FIX_INTEREST_AREA_ID.values, None, sub_df.CURRENT_FIX_DURATION.values

			sp_ordinal_pos = sp_word_pos.astype(int)
			SP_ordinal_pos.append(sp_ordinal_pos)
			SP_fix_dur.append(sp_fix_dur)
			SP_landing_pos.append(sp_fix_loc)

			sp_token = [sn_str.split()[int(i)+first-1] for i in sp_ordinal_pos]
			sp_token_str = '[CLS]' + ' ' + ' '.join(sp_token) + ' ' + '[SEP]'

			#tokenization and padding for scanpath, i.e. fixated word sequence
			sp_tokens = tokenizer.encode_plus(sp_token_str.split(),
											add_special_tokens = False,
											truncation=True,
											max_length = cf['max_sp_token'],
											padding = 'max_length',
											return_attention_mask=True,
											is_split_into_words=True)
			encoded_sp = sp_tokens['input_ids']
			mask_sp = sp_tokens['attention_mask']
			#index start from 0, CLS -> 0 and SEP -> last index
			word_ids_sp = sp_tokens.word_ids()
			word_ids_sp = [val if val is not None else np.nan for val in word_ids_sp]
			SP_input_ids.append(encoded_sp)
			SP_attention_mask.append(mask_sp)
			WORD_ids_sp.append(word_ids_sp)

			#sentence information
			SN_input_ids.append(encoded_sn)#TODO index if not by_page
			SN_attention_mask.append(mask_sn)#TODO same
			SN_WORD_len.append(sn_word_len)
			WORD_ids_sn.append(word_ids_sn)
			sub_id_list.append(sub_id)
			SN_newlines.append(sn_newlines)

	#padding for batch computation
	SP_ordinal_pos = pad_seq(SP_ordinal_pos, max_len=(cf["max_sp_len"]), pad_value=cf["max_sn_len"])#if you change the pad value, modify prepare_scanpath accordingly
	SP_fix_dur = pad_seq(SP_fix_dur, max_len=(cf["max_sp_len"]), pad_value=0)
	SP_landing_pos = pad_seq(SP_landing_pos, cf["max_sp_len"], pad_value=0, dtype=np.float32)
	SN_WORD_len = pad_seq_with_nan(SN_WORD_len, cf["max_sn_len"], dtype=np.float32)

	#assign type
	SN_input_ids = np.asarray(SN_input_ids, dtype=np.int64)
	SN_attention_mask = np.asarray(SN_attention_mask, dtype=np.float32)
	SP_input_ids = np.asarray(SP_input_ids, dtype=np.int64)
	SP_attention_mask = np.asarray(SP_attention_mask, dtype=np.float32)
	sub_id_list = np.asarray(sub_id_list, dtype=np.int64)
	WORD_ids_sn = np.asarray(WORD_ids_sn)
	WORD_ids_sp = np.asarray(WORD_ids_sp)
	SN_newlines = np.asarray(SN_newlines)
	#print(SN_newlines)

	data = {
		"SN_input_ids": SN_input_ids,
		"SN_attention_mask": SN_attention_mask,
		"SN_WORD_len": SN_WORD_len,
		"WORD_ids_sn": WORD_ids_sn,
		"SP_input_ids": SP_input_ids,
		"SP_attention_mask": SP_attention_mask,
		"WORD_ids_sp": WORD_ids_sp,
		"SP_ordinal_pos": np.array(SP_ordinal_pos),
		"SP_landing_pos": np.array(SP_landing_pos),
		"SP_fix_dur": np.array(SP_fix_dur),
		"sub_id": sub_id_list,
		"SN_newlines": SN_newlines
	}
	#print(data["SP_ordinal_pos"])

	return data

def compute_BSC_word_length(sn_df):
	word_len = sn_df.LEN.values
	wl_list = []
	for wl in word_len:
		wl_list.extend([wl]*wl)
	arr = np.asarray(wl_list, dtype=np.float32)
	#length of a punctuation is 0, plus an epsilon to avoid division output inf
	arr[arr==0] = 1/(0+0.5)
	arr[arr!=0] = 1/(arr[arr!=0])
	return arr

def pad_seq(seqs, max_len, pad_value, dtype=np.compat.long):
	padded = np.full((len(seqs), max_len), fill_value=pad_value, dtype=dtype)
	for i, seq in enumerate(seqs):
		if np.any(seq==None):
			padded[i,] = 0
			continue
		padded[i, 0] = 0
		padded[i, 1:(len(seq)+1)] = seq
		if pad_value !=0:
			padded[i, len(seq)+1] = pad_value -1

	return padded

def pad_seq_with_nan(seqs, max_len, dtype=np.compat.long):
	padded = np.full((len(seqs), max_len), fill_value=np.nan, dtype=dtype)
	for i, seq in enumerate(seqs):
		padded[i, 1:(len(seq)+1)] = seq
	return padded

def _process_BSC_corpus(sn_list, reader_list, word_info_df, eyemovement_df, tokenizer, cf):
	"""
	SN_token_embedding   <CLS>, bla, bla, <SEP>
	SP_token_embedding   <CLS>, bla, bla, <SEP>
	SP_ordinal_pos 0, bla, bla, max_sp_len
	SP_fix_dur     0, bla, bla, 0
	SN_len         original sentence length without start and end tokens
	"""
	SN_input_ids, SN_attention_mask, SN_WORD_len = [], [], []
	SP_input_ids, SP_attention_mask = [], []
	SP_ordinal_pos, SP_landing_pos, SP_fix_dur = [], [], []
	sub_id_list = []
	for sn_id in sn_list:
		#print('sub_id:', sub_id)
		#process sentence sequence
		sn_df = eyemovement_df[eyemovement_df.sn==sn_id]
		sn = word_info_df[word_info_df.SN == sn_id]
		sn_str = ''.join(sn.WORD.values)
		sn_word_len = compute_BSC_word_length(sn)

		#tokenization and padding
		tokenizer.padding_side = 'right'
		tokens = tokenizer.encode_plus(sn_str,
										add_special_tokens = True,
										truncation=True,
										max_length = cf["max_sn_len"],
										padding = 'max_length',
										return_attention_mask=True)
		encoded_sn = tokens["input_ids"]
		mask_sn = tokens["attention_mask"]

		#process fixation sequence
		for sub_id in reader_list:
			sub_df = sn_df[sn_df.id==sub_id]
			if len(sub_df) == 0:
				#no scanpath data found for the subject
				continue

			#last fixation go back to the first character with fl = 0 -- seems to be outlier point? remove
			if sub_df.iloc[-1].wn == 1 and sub_df.iloc[-1].fl == 0:
				sub_df = sub_df.iloc[:-1]

			sp_word_pos, sp_fix_loc, sp_fix_dur = sub_df.wn.values, sub_df.fl.values, sub_df.dur.values
			sp_landing_pos_char = np.modf(sp_fix_loc)[0]
			SP_landing_pos.append(sp_landing_pos_char)

			#Convert word-based ordinal positions to token(character)-based ordinal positions
			#When the fixated word index is less than 0, set it to 0
			sp_fix_loc = np.where(sp_fix_loc<0, 0, sp_fix_loc)
			sp_ordinal_pos = [np.sum(sn[sn.NW<value].LEN) + np.ceil(sp_fix_loc[count]+ 1e-10) for count, value in enumerate(sp_word_pos)]
			SP_ordinal_pos.append(sp_ordinal_pos)
			SP_fix_dur.append(sp_fix_dur)

			#tokenization and padding for scanpath, i.e. fixated word sequence
			sp_token = [sn_str[int(i-1)] for i in sp_ordinal_pos]
			sp_token_str = '[CLS]' + ''.join(sp_token) + '[SEP]'
			sp_tokens = tokenizer.encode_plus(sp_token_str,
												add_special_tokens = False,
												truncation=True,
												max_length = cf["max_sp_len"],
												padding = 'max_length',
												return_attention_mask=True)
			encoded_sp = sp_tokens["input_ids"]
			mask_sp = sp_tokens["attention_mask"]
			SP_input_ids.append(encoded_sp)
			SP_attention_mask.append(mask_sp)

			#sentence information
			SN_input_ids.append(encoded_sn)
			SN_attention_mask.append(mask_sn)
			SN_WORD_len.append(sn_word_len)
			sub_id_list.append(sub_id)

	#padding for batch computation
	SP_ordinal_pos = pad_seq(SP_ordinal_pos, max_len=(cf["max_sp_len"]), pad_value=cf["max_sn_len"])
	SP_fix_dur = pad_seq(SP_fix_dur, max_len=(cf["max_sp_len"]), pad_value=0)
	SP_landing_pos = pad_seq(SP_landing_pos, cf["max_sp_len"], pad_value=0, dtype=np.float32)
	SN_WORD_len = pad_seq_with_nan(SN_WORD_len, cf["max_sn_len"], dtype=np.float32)

	#assign type
	SN_input_ids = np.asarray(SN_input_ids, dtype=np.int64)
	SN_attention_mask = np.asarray(SN_attention_mask, dtype=np.float32)
	SP_input_ids = np.asarray(SP_input_ids, dtype=np.int64)
	SP_attention_mask = np.asarray(SP_attention_mask, dtype=np.float32)
	sub_id_list = np.asarray(sub_id_list, dtype=np.int64)

	data = {"SN_input_ids": SN_input_ids, "SN_attention_mask": SN_attention_mask, "SN_WORD_len": SN_WORD_len,
			"SP_input_ids": SP_input_ids, "SP_attention_mask": SP_attention_mask,
			"SP_ordinal_pos": np.array(SP_ordinal_pos), "SP_landing_pos": np.array(SP_landing_pos), "SP_fix_dur": np.array(SP_fix_dur),
			"sub_id": sub_id_list}

	return data

class BSCdataset(Dataset):
	"""Return BSC dataset."""

	def __init__(
		self,
		word_info_df, eyemovement_df, cf, reader_list, sn_list, tokenizer
	):
		self.data = _process_BSC_corpus(sn_list, reader_list, word_info_df, eyemovement_df, tokenizer, cf)

	def __len__(self):
		return len(self.data["SN_input_ids"])

	def __getitem__(self,idx):
		sample = {}
		sample["sn_input_ids"] = self.data["SN_input_ids"][idx,:]
		sample["sn_attention_mask"] = self.data["SN_attention_mask"][idx,:]
		sample["sn_word_len"] = self.data['SN_WORD_len'][idx,:]

		sample["sp_input_ids"] = self.data["SP_input_ids"][idx,:]
		sample["sp_attention_mask"] = self.data["SP_attention_mask"][idx,:]

		sample["sp_pos"] = self.data["SP_ordinal_pos"][idx,:]
		sample["sp_fix_dur"] = self.data["SP_fix_dur"][idx,:]
		sample["sp_landing_pos"] = self.data["SP_landing_pos"][idx,:]

		sample["sub_id"] = self.data["sub_id"][idx]

		return sample

def calculate_mean_std(dataloader, feat_key, padding_value=0, scale=1):
	#calculate mean
	total_sum = 0
	total_num = 0
	for batchh in dataloader:
		batchh.keys()
		feat = batchh[feat_key]/scale
		feat = torch.nan_to_num(feat)
		total_num += len(feat.view(-1).nonzero())
		total_sum += feat.sum()
	feat_mean = total_sum / total_num
	#calculate std
	sum_of_squared_error = 0
	for batchh in dataloader:
		batchh.keys()
		feat = batchh[feat_key]/scale
		feat = torch.nan_to_num(feat)
		mask = ~torch.eq(feat, padding_value)
		sum_of_squared_error += (((feat - feat_mean).pow(2))*mask).sum()
	feat_std = torch.sqrt(sum_of_squared_error / total_num)
	return feat_mean, feat_std

def load_label(sp_pos, cf, labelencoder, device):
	#prepare label and mask
	pad_mask = torch.eq(sp_pos[:, 1:], cf["max_sn_len"])
	end_mask = torch.eq(sp_pos[:, 1:], cf["max_sn_len"]-1)
	mask = pad_mask + end_mask
	sac_amp = sp_pos[:, 1:] - sp_pos[:, :-1]
	label = sp_pos[:, 1:]*mask + sac_amp*~mask
	label = torch.where(label>cf["max_saccade_len"]-1, cf["max_saccade_len"]-1, label).to('cpu').detach()
	label = torch.where(label<-cf["max_saccade_len"]+3, -cf["max_saccade_len"]+3, label).to('cpu').detach().numpy()
	label = labelencoder.transform(label.reshape(-1)).reshape(label.shape[0], label.shape[1])
	pad_mask = pad_mask.to('cpu').detach().numpy()
	label = torch.from_numpy(label).to(device)
	return pad_mask, label


def likelihood(pred, label, mask):
	#test
	#res = F.nll_loss(torch.tensor(pred), torch.tensor(label))
	label = one_hot_encode(label, pred.shape[1])
	res = np.sum(np.multiply(pred, label), axis=1)
	res = np.sum(res * np.array(~mask))/np.sum(np.array(~mask))
	return res

def eval_log_llh(dnn_out, label, pad_mask):
	res = []
	dnn_out = np.log2(dnn_out + 1e-10)
	#For each scanpath calculate the likelihood and then find the average
	for sp_indx in range(dnn_out.shape[0]):
		out = likelihood(dnn_out[sp_indx, :, :], label[sp_indx, :], pad_mask[sp_indx, :])
		res.append(out)

	return res


def prepare_scanpath(sp_dnn, sn_len, sp_human, cf):
	max_sp_len = sp_dnn.shape[1]
	sp_human = sp_human.detach().to('cpu').numpy()

	#stop_indx = [np.where(sp_dnn[i,:]==(sn_len[i]+1))[0][0] for i in range(sp_dnn.shape[0])]
	#Find the number "sn_len+1" -> the end point
	stop_indx = []
	for i in range(sp_dnn.shape[0]):
		stop = np.where(sp_dnn[i,:]==(sn_len[i]+1))[0]
		if len(stop)==0:#no end point can be find -> exceeds the maximum length of the generated scanpath
			stop_indx.append(max_sp_len-1)
		else:
			stop_indx.append(stop[0])

	#Truncating data after the end point
	sp_dnn_cut = [sp_dnn[i][:stop_indx[i]+1] for i in range(sp_dnn.shape[0])]
	#replace the last teminal number to cf["max_sn_len"]-1, keep the same as the human scanpath label
	for i in range(len(sp_dnn_cut)):
		sp_dnn_cut[i][-1] = cf["max_sn_len"]-1

	#process the human scanpath data, truncating data after the end point
	stop_indx = [np.where(sp_human[i,:]==cf["max_sn_len"]-1)[0][0] for i in range(sp_human.shape[0])]
	sp_human_cut = [sp_human[i][:stop_indx[i]+1] for i in range(sp_human.shape[0])]
	return sp_dnn_cut, sp_human_cut



def celer_load_native_speaker():
	sub_metadata_path = './Data/celer/metadata.tsv'
	sub_infor = pd.read_csv(sub_metadata_path, delimiter='\t')
	native_sub_list = sub_infor[sub_infor.L1 == 'English'].List.values
	return native_sub_list.tolist()

def compute_word_length_celer(arr):
	#length of a punctuation is 0, plus an epsilon to avoid division output inf
	arr = arr.astype('float64')
	arr[arr==0] = 1/(0+0.5)
	arr[arr!=0] = 1/(arr[arr!=0])
	return arr
def _process_celer(sn_list, reader_list, word_info_df, eyemovement_df, tokenizer, cf):
	"""
	SN_token_embedding   <CLS>, bla, bla, <SEP>
	SP_token_embedding       <CLS>, bla, bla, <SEP>
	SP_ordinal_pos 0, bla, bla, max_sp_len
	SP_fix_dur     0, bla, bla, 0
	"""
	SN_input_ids, SN_attention_mask, SN_WORD_len, WORD_ids_sn = [], [], [], []
	SP_input_ids, SP_attention_mask, WORD_ids_sp = [], [], []
	SP_ordinal_pos, SP_landing_pos, SP_fix_dur = [], [], []
	sub_id_list =  []
	for sn_id in tqdm(sn_list):
		#process sentence sequence
		sn_df = eyemovement_df[eyemovement_df.sentenceid==sn_id]
		#notice: Each sentence is recorded multiple times in file |word_info_df|.
		sn = word_info_df[word_info_df.sentenceid == sn_id]
		sn = sn[sn['list']==sn.list.values.tolist()[0]]
		#compute word length for each word
		sn_word_len = compute_word_length_celer(sn.WORD_LEN.values)

		sn_str = sn.sentence.iloc[-1]
		#nessacery sanity check, when split sentence to words, the length of sentence should match the sentence length recorded in celer dataset
		if sn_id in ['1987/w7_019/w7_019.295-3', '1987/w7_036/w7_036.147-43', '1987/w7_091/w7_091.360-6']:
			#extra inverted commas at the end of the sentence
			sn_str = sn_str[:-3] + sn_str[-1:]
		if sn_id == '1987/w7_085/w7_085.200-18':
			sn_str = sn_str[:43] + sn_str[44:]
		sn_len = len(sn_str.split())

		#tokenization and padding
		tokenizer.padding_side = 'right'
		sn_str = '[CLS]' + ' ' + sn_str + ' ' + '[SEP]'
		#pre-tokenized input
		tokens = tokenizer.encode_plus(sn_str.split(),
										add_special_tokens = False,
										truncation=False,
										max_length = cf['max_sn_token'],
										padding = 'max_length',
										return_attention_mask=True,
										is_split_into_words=True)
		encoded_sn = tokens['input_ids']
		mask_sn = tokens['attention_mask']
		#use offset mapping to determine if two tokens are in the same word.
		#index start from 0, CLS -> 0 and SEP -> last index
		word_ids_sn = tokens.word_ids()
		word_ids_sn = [val if val is not None else np.nan for val in word_ids_sn]

		#process fixation sequence
		for sub_id in reader_list:
			sub_df = sn_df[sn_df.list==sub_id]
			# remove fixations on non-words
			sub_df = sub_df.loc[sub_df.CURRENT_FIX_INTEREST_AREA_LABEL != '.']
			if len(sub_df) == 0:
				#no scanpath data found for the subject
				continue

			#prepare decoder input and output
			sp_word_pos, sp_fix_loc, sp_fix_dur = sub_df.CURRENT_FIX_INTEREST_AREA_ID.values, sub_df.CURRENT_FIX_NEAREST_INTEREST_AREA_DISTANCE.values, sub_df.CURRENT_FIX_DURATION.values

			#dataset is noisy -> sanity check
			# 1) check if recorded fixation duration are within reasonable limits
			#Less than 50ms attempt to merge with neighbouring fixation if fixate is on the same word, otherwise delete
			outlier_indx = np.where(sp_fix_dur<50)[0]
			if outlier_indx.size>0:
				for out_idx in range(len(outlier_indx)):
					outlier_i = outlier_indx[out_idx]
					merge_flag = False

					#outliers are commonly found in the fixation of the last record and the first record, and are removed directly
					if outlier_i == len(sp_fix_dur)-1 or outlier_i == 0:
						merge_flag = True

					else:
						if outlier_i-1 >= 0 and merge_flag == False:
							#try to merge with the left fixation
							if sub_df.iloc[outlier_i].CURRENT_FIX_INTEREST_AREA_LABEL == sub_df.iloc[outlier_i-1].CURRENT_FIX_INTEREST_AREA_LABEL:
								sp_fix_dur[outlier_i-1] = sp_fix_dur[outlier_i-1] + sp_fix_dur[outlier_i]
								merge_flag = True

						if outlier_i+1 < len(sp_fix_dur) and merge_flag == False:
							#try to merge with the right fixation
							if sub_df.iloc[outlier_i].CURRENT_FIX_INTEREST_AREA_LABEL == sub_df.iloc[outlier_i+1].CURRENT_FIX_INTEREST_AREA_LABEL:
								sp_fix_dur[outlier_i+1] = sp_fix_dur[outlier_i+1] + sp_fix_dur[outlier_i]
								merge_flag = True

					sp_word_pos = np.delete(sp_word_pos, outlier_i)
					sp_fix_loc = np.delete(sp_fix_loc, outlier_i)
					sp_fix_dur = np.delete(sp_fix_dur, outlier_i)
					sub_df.drop(sub_df.index[outlier_i], axis=0, inplace=True)
					outlier_indx = outlier_indx-1

			# 2) scanpath too long, remove outliers, speed up the inference
			if len(sp_word_pos) > 50: # 72/10684
				continue
			# 3)scanpath too short for a normal length sentence
			if len(sp_word_pos)<=1 and sn_len>10:
				continue

			# 4) check landing position feature
			#assign missing value to 'nan'
			sp_fix_loc=np.where(sp_fix_loc=='.', np.nan, sp_fix_loc)
			#convert string of number of float type
			sp_fix_loc = [float(i) for i in sp_fix_loc]

			#Outliers in calculated landing positions due to lack of valid AOI data, assign to 'nan'
			if np.nanmax(sp_fix_loc)>35:
				missing_idx = np.where(np.array(sp_fix_loc)>5)[0]
				for miss in missing_idx:
					if sub_df.iloc[miss].CURRENT_FIX_INTEREST_AREA_LEFT in ['NONE', 'BEFORE', 'AFTER', 'BOTH']:
						sp_fix_loc[miss] = np.nan
					else:
						print('Landing position calculation error. Unknown cause, needs to be checked')

			sp_ordinal_pos = sp_word_pos.astype(int)
			SP_ordinal_pos.append(sp_ordinal_pos)
			SP_fix_dur.append(sp_fix_dur)
			SP_landing_pos.append(sp_fix_loc)

			sp_token = [sn_str.split()[int(i)] for i in sp_ordinal_pos]
			sp_token_str = '[CLS]' + ' ' + ' '.join(sp_token) + ' ' + '[SEP]'

			#tokenization and padding for scanpath, i.e. fixated word sequence
			sp_tokens = tokenizer.encode_plus(sp_token_str.split(),
											add_special_tokens = False,
											truncation=False,
											max_length = cf['max_sp_token'],
											padding = 'max_length',
											return_attention_mask=True,
											is_split_into_words=True)
			encoded_sp = sp_tokens['input_ids']
			mask_sp = sp_tokens['attention_mask']
			#index start from 0, CLS -> 0 and SEP -> last index
			word_ids_sp = sp_tokens.word_ids()
			word_ids_sp = [val if val is not None else np.nan for val in word_ids_sp]
			SP_input_ids.append(encoded_sp)
			SP_attention_mask.append(mask_sp)
			WORD_ids_sp.append(word_ids_sp)

			#sentence information
			SN_input_ids.append(encoded_sn)
			SN_attention_mask.append(mask_sn)
			SN_WORD_len.append(sn_word_len)
			WORD_ids_sn.append(word_ids_sn)
			sub_id_list.append(int(sub_id))

	#padding for batch computation
	SP_ordinal_pos = pad_seq(SP_ordinal_pos, max_len=(cf["max_sp_len"]), pad_value=cf["max_sn_len"])
	SP_fix_dur = pad_seq(SP_fix_dur, max_len=(cf["max_sp_len"]), pad_value=0)
	SP_landing_pos = pad_seq(SP_landing_pos, cf["max_sp_len"], pad_value=0, dtype=np.float32)
	SN_WORD_len = pad_seq_with_nan(SN_WORD_len, cf["max_sn_len"], dtype=np.float32)

	#assign type
	SN_input_ids = np.asarray(SN_input_ids, dtype=np.int64)
	SN_attention_mask = np.asarray(SN_attention_mask, dtype=np.float32)
	SP_input_ids = np.asarray(SP_input_ids, dtype=np.int64)
	SP_attention_mask = np.asarray(SP_attention_mask, dtype=np.float32)
	sub_id_list = np.asarray(sub_id_list, dtype=np.int64)
	WORD_ids_sn = np.asarray(WORD_ids_sn)
	WORD_ids_sp = np.asarray(WORD_ids_sp)

	data = {"SN_input_ids": SN_input_ids, "SN_attention_mask": SN_attention_mask, "SN_WORD_len": SN_WORD_len, "WORD_ids_sn": WORD_ids_sn,
	 		"SP_input_ids": SP_input_ids, "SP_attention_mask": SP_attention_mask, "WORD_ids_sp": WORD_ids_sp,
			"SP_ordinal_pos": np.array(SP_ordinal_pos), "SP_landing_pos": np.array(SP_landing_pos), "SP_fix_dur": np.array(SP_fix_dur),
			"sub_id": sub_id_list,
			}

	return data

class celerdataset(Dataset):
	"""Return celer dataset."""

	def __init__(
		self,
		word_info_df, eyemovement_df, cf, reader_list, sn_list, tokenizer
	):

		self.data = _process_celer(sn_list, reader_list, word_info_df, eyemovement_df, tokenizer, cf)

	def __len__(self):
		return len(self.data["SN_input_ids"])


	def __getitem__(self,idx):
		sample = {}
		sample["sn_input_ids"] = self.data["SN_input_ids"][idx,:]
		sample["sn_attention_mask"] = self.data["SN_attention_mask"][idx,:]
		sample["sn_word_len"] = self.data['SN_WORD_len'][idx,:]
		sample['word_ids_sn'] =  self.data['WORD_ids_sn'][idx,:]

		sample["sp_input_ids"] = self.data["SP_input_ids"][idx,:]
		sample["sp_attention_mask"] = self.data["SP_attention_mask"][idx,:]
		sample['word_ids_sp'] =  self.data['WORD_ids_sp'][idx,:]

		sample["sp_pos"] = self.data["SP_ordinal_pos"][idx,:]
		sample["sp_fix_dur"] = self.data["SP_fix_dur"][idx,:]
		sample["sp_landing_pos"] = self.data["SP_landing_pos"][idx,:]

		sample["sub_id"] = self.data["sub_id"][idx]

		return sample


def one_hot_encode(arr, dim):
	# one hot encode
	onehot_encoded = np.zeros((arr.shape[0], dim))
	for idx, value in enumerate(arr):
		onehot_encoded[idx, value] = 1

	return onehot_encoded



def gradient_clipping(dnn_model, clip = 10):
	torch.nn.utils.clip_grad_norm_(dnn_model.parameters(),clip)



def _process_meco(sn_df, data_df, cf, reader_list, sn_list, tokenizer):
	# Initialize lists to store processed data
	SN_input_ids, SN_attention_mask, SN_WORD_len, WORD_ids_sn = [], [], [], []
	SP_input_ids, SP_attention_mask, WORD_ids_sp = [], [], []
	SP_ordinal_pos, SP_landing_pos, SP_fix_dur = [], [], []
	sub_id_list, raw_split_words = [], []
	SN_newlines = [] #NEW

	# We need to iterate over the subjects and then over the trialid or itemid
	for sub_id in tqdm(reader_list):
		lang_code = sub_id.split('_')[0]
		#if lang_code not in ['du', 'en', 'ee', 'fi', 'ge', 'it', 'no', 'es', 'tr']:
		# if lang_code not in ["en"]:
		# 	continue
		# Iterate over sentences in the MeCo dataset
		for sent_id in sn_list:
			# Extract text by language rowID and column textID
			text = sn_df.loc[sn_df['Language'] == lang_code, sent_id].iloc[0].replace('"', '')#TODO
			#print(sent_id)
			#print(text[:100])
			subj_fix_df = data_df[(data_df['uniform_id'] == sub_id)\
       			& (data_df['itemid'] == sent_id)]
			if len(subj_fix_df) == 0:
				continue
			# Processing sentence sequence
			text_word_len = [len(word) for word in text.split()]  # Assuming word length calculation is similar

			# Tokenization and padding for sentence
			text_str = '[CLS] ' + text + ' [SEP]'
			#NEW
			line_list = text_str.splitlines()
			#print(line_list)
			line_len = np.array([len(line.split()) for line in line_list])
			#print(line_len)
			cum_line_len = np.cumsum(line_len[:-1])#linebreak not relevant after last line
			#print(cum_line_len)
			sn_newlines = np.full(cf["max_sn_len"], False)
			for i in cum_line_len:
				sn_newlines[i] = True
			#print(sn_newlines)

			tokens = tokenizer.encode_plus(text_str.split(),
										add_special_tokens=False,
										truncation=True, # Worst case truncate
										max_length=cf['max_sn_token'],
										padding='max_length',
										return_attention_mask=True,
										is_split_into_words=True)

			encoded_sn = tokens['input_ids']
			mask_sn = tokens['attention_mask']
			word_ids_sn = tokens.word_ids()
			word_ids_sn = [val if val is not None else np.nan for val in word_ids_sn]

			# Process fixation sequence for each subject using uniform_id instead of subid
			# Extract relevant columns for fixation processing
			sp_word_pos = subj_fix_df['wordnum'].dropna().values
			sp_fix_dur = subj_fix_df['dur'].dropna().values
			sp_landing_pos = subj_fix_df['word.land'].dropna().values
			sp_words = subj_fix_df['word'].dropna().values.tolist()
			#print("word: ", sp_words[:20])
			split_text = re.split(r'(?<=-)\b|\s+', text_str)
			raw_slit_text = re.split(r'(?<=-)\b|\s+', text)
			# Processing for scanpath (fixated word sequence)
			try:
				sp_token = [split_text[int(i)] for i in sp_word_pos]
			except IndexError as e:
				print(f"Index Error: {e}")
				continue

			sp_token_str = '[CLS] ' + ' '.join(sp_token) + ' [SEP]'

			# Tokenization and padding for scanpath
			sp_tokens = tokenizer.encode_plus(sp_token_str.split(),
											add_special_tokens=False,
											truncation=True, # Worst case truncate
											max_length=cf['max_sp_token'],
											padding='max_length',
											return_attention_mask=True,
											is_split_into_words=True)
			encoded_sp = sp_tokens['input_ids']
			mask_sp = sp_tokens['attention_mask']
			word_ids_sp = sp_tokens.word_ids()
			word_ids_sp = [val if val is not None else np.nan for val in word_ids_sp]
			if len(sp_landing_pos) > cf['max_sp_len']:
				print("Scanpath length exceeds max length")
				continue
			elif len(word_ids_sp) > cf['max_sp_token']:
				print("Scanpath token length exceeds max length")
				continue
			elif len(word_ids_sn) > cf['max_sn_token']:
				print("Sentence length exceeds max length")
				continue
			elif len(word_ids_sn) > cf['max_sn_token']:
				print("Sentence token length exceeds max length")
				continue
			# Append processed data to lists
			SP_ordinal_pos.append(sp_word_pos)
			SP_fix_dur.append(sp_fix_dur)
			SP_input_ids.append(encoded_sp)
			SP_attention_mask.append(mask_sp)
			WORD_ids_sp.append(word_ids_sp)
			SN_input_ids.append(encoded_sn)
			SN_attention_mask.append(mask_sn)
			SN_WORD_len.append(text_word_len)
			WORD_ids_sn.append(word_ids_sn)
			sub_id_list.append(sub_id)
			SP_landing_pos.append(sp_landing_pos)

			raw_split_words.append(raw_slit_text)

			SN_newlines.append(sn_newlines)


	# Padding for batch computation 
	raw_sp_pos = SP_ordinal_pos

	SP_ordinal_pos = pad_seq(SP_ordinal_pos, max_len=(cf["max_sp_len"]), pad_value=cf["max_sn_len"])
	SP_fix_dur = pad_seq(SP_fix_dur, max_len=(cf["max_sp_len"]), pad_value=0)
	SP_landing_pos = pad_seq(SP_landing_pos, cf["max_sp_len"], pad_value=0, dtype=np.float32)
	SN_WORD_len = pad_seq_with_nan(SN_WORD_len, cf["max_sn_len"], dtype=np.float32)

	#assign type
	SN_input_ids = np.asarray(SN_input_ids, dtype=np.int64)
	SN_attention_mask = np.asarray(SN_attention_mask, dtype=np.float32)
	SP_input_ids = np.asarray(SP_input_ids, dtype=np.int64)
	SP_attention_mask = np.asarray(SP_attention_mask, dtype=np.float32)

	sub_id_list = np.asarray(sub_id_list)
	WORD_ids_sn = np.asarray(WORD_ids_sn)
	WORD_ids_sp = np.asarray(WORD_ids_sp)

	SN_newlines = np.asarray(SN_newlines)

	# Create the final data structure
	data = {
		"SN_input_ids": np.array(SN_input_ids),
		"SN_attention_mask": np.array(SN_attention_mask),
		"SN_WORD_len": np.array(SN_WORD_len),
		"WORD_ids_sn": np.array(WORD_ids_sn),
		"SP_input_ids": np.array(SP_input_ids),
		"SP_attention_mask": np.array(SP_attention_mask),
		"WORD_ids_sp": np.array(WORD_ids_sp),
		"SP_ordinal_pos": np.array(SP_ordinal_pos),
		"SP_fix_dur": np.array(SP_fix_dur),
		"sub_id": np.array(sub_id_list),
		"SP_landing_pos": np.array(SP_landing_pos),
		"raw_split_words": raw_split_words,
		"raw_sp_pos": raw_sp_pos,
		"SN_newlines": SN_newlines,
	}

	return data

class mecodataset(Dataset):
	"""Return celer dataset."""
	def __init__(self, sn_df, data_df, cf, reader_list, sn_list, tokenizer):
		self.data = _process_meco(sn_df, data_df, cf, reader_list, sn_list, tokenizer)

	def __len__(self):
		return len(self.data["SN_input_ids"])


	def __getitem__(self,idx):
		sample = {}
		sample["sn_input_ids"] = self.data["SN_input_ids"][idx,:]
		sample["sn_attention_mask"] = self.data["SN_attention_mask"][idx,:]
		sample["sn_word_len"] = self.data['SN_WORD_len'][idx,:]
		sample['word_ids_sn'] =  self.data['WORD_ids_sn'][idx,:]

		sample["sp_input_ids"] = self.data["SP_input_ids"][idx,:]
		sample["sp_attention_mask"] = self.data["SP_attention_mask"][idx,:]
		sample['word_ids_sp'] =  self.data['WORD_ids_sp'][idx,:]

		sample["sp_pos"] = self.data["SP_ordinal_pos"][idx,:]
		sample["sp_fix_dur"] = self.data["SP_fix_dur"][idx,:]
		sample["sp_landing_pos"] = self.data["SP_landing_pos"][idx,:]

		sample["sub_id"] = self.data["sub_id"][idx]

		#NEW: word indices BEFORE which a newline occurs
		sample["sn_newlines"] = self.data["SN_newlines"][idx,:]

		return sample
