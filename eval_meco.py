import argparse
import numpy as np
import os
import pickle

from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast

from model import Eyettention
from utils import *

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

#initialisations
sp_dnn_list = []
sp_human_list = []
text_list = []

loss_dict = {}
loss_dict['test_ll'] = []

#load model
cf = {"model_pretrained": "bert-base-cased",
			"lr": 1e-3,
			"max_grad_norm": 10,
			"n_epochs": 1000,
			#"n_folds": 5,
			"dataset": 'meco',
			"atten_type": args.atten_type,
			"batch_size": 32,#TODO
			#following 4 lines CHANGED. Can't get more than 512 tokens into BERT!#TODO
			"max_sn_len": 201, #at least 201 for meco
            "max_saccade_len": 148,#same architecture as the trained model
			"max_sn_token": 183, #maximum number of tokens a sentence includes. include start token and end token
			"max_sp_len": 396, #max number of words in a scanpath, include start token and end token #or 1130? #needs to be the actual sp len + 2 for the CLS
			"max_sp_token": 512, #maximum number of tokens a scanpath includes. include start token and end token UNNECESSARY
			"norm_type": 'z-score',
			"earlystop_patience": 20,
			"max_pred_len":args.max_pred_len,
			"device":device
			}

dnn = Eyettention(cf)

with open('{}/res_SAT_{}_eyettention_{}.pickle'.format(args.save_data_folder, args.test_mode, args.atten_type), 'rb') as handle:
	ld = pickle.load(handle)
     
#Encode the label into interger categories, setting the exclusive category 'cf["max_sn_len"]-1' as the end sign
le = LabelEncoder()
le.fit(np.append(np.arange(-cf["max_saccade_len"]+3, cf["max_saccade_len"]-1), cf["max_saccade_len"]-1))

#eval data
data_df, sn_df, reader_list = load_corpus(cf["dataset"])
sn_list = [col for col in sn_df.columns if col != 'Language']

tokenizer = BertTokenizerFast.from_pretrained(cf['model_pretrained'])

dataset_test = mecodataset(sn_df, data_df, cf, reader_list, sn_list, tokenizer)
test_dataloaderr = DataLoader(dataset_test, batch_size = cf["batch_size"], shuffle = False, drop_last=False)

fix_dur_mean, fix_dur_std = ld['fix_dur_mean'], ld['fix_dur_std']
landing_pos_mean, landing_pos_std = ld['landing_pos_mean'], ld['landing_pos_std']
sn_word_len_mean, sn_word_len_std = ld['sn_word_len_mean'], ld['sn_word_len_std']

#evaluation
dnn.eval()
res_llh=[]
dnn.load_state_dict(torch.load(os.path.join(args.save_data_folder,f'CELoss_SAT_{args.test_mode}_eyettention_{args.atten_type}.pth'), map_location='cpu'))
dnn.to(device)
batch_indx = 0
for batchh in test_dataloaderr:
    print("evaluating batch nr.", batch_indx)
    with torch.no_grad():
        sn_input_ids_test = batchh["sn_input_ids"].to(device)
        sn_attention_mask_test = batchh["sn_attention_mask"].to(device)
        word_ids_sn_test = batchh["word_ids_sn"].to(device)
        sn_word_len_test = batchh["sn_word_len"].to(device)

        sp_input_ids_test = batchh["sp_input_ids"].to(device)
        sp_attention_mask_test = batchh["sp_attention_mask"].to(device)
        word_ids_sp_test = batchh["word_ids_sp"].to(device)

        sp_pos_test = batchh["sp_pos"].to(device)
        sp_landing_pos_test = batchh["sp_landing_pos"].to(device)
        sp_fix_dur_test = (batchh["sp_fix_dur"]/1000).to(device)

        sn_newlines_test = batchh["sn_newlines"].to(device)

        #reconstruct text for reference, including newlines
        keyword = tokenizer.decode(sn_input_ids_test[2])#first actual word after cls and perhaps a function word - works only for this dataset!
        text_with_newlines = next(text for text in sn_list if text.split()[1]==keyword)

        #normalize gaze features
        mask = ~torch.eq(sp_fix_dur_test, 0)
        sp_fix_dur_test = (sp_fix_dur_test-fix_dur_mean)/fix_dur_std * mask
        sp_landing_pos_test = (sp_landing_pos_test - landing_pos_mean)/landing_pos_std * mask
        sp_fix_dur_test = torch.nan_to_num(sp_fix_dur_test)
        sp_landing_pos_test = torch.nan_to_num(sp_landing_pos_test)
        sn_word_len_test = (sn_word_len_test - sn_word_len_mean)/sn_word_len_std
        sn_word_len_test = torch.nan_to_num(sn_word_len_test)

        dnn_out_test, atten_weights_test = dnn(sn_emd=sn_input_ids_test,
                                                sn_mask=sn_attention_mask_test,
                                                sp_emd=sp_input_ids_test,
                                                sp_pos=sp_pos_test,
                                                word_ids_sn=word_ids_sn_test,
                                                word_ids_sp=word_ids_sp_test,
                                                sp_fix_dur=sp_fix_dur_test,
                                                sp_landing_pos=sp_landing_pos_test,
                                                sn_word_len = sn_word_len_test,
                                                sn_newlines = sn_newlines_test)

        #We do not use nn.CrossEntropyLoss here to calculate the likelihood because it combines nn.LogSoftmax and nn.NLL,
        #while nn.LogSoftmax returns a log value based on e, we want 2 instead
        #m = nn.LogSoftmax(dim=2) -- base e, we want base 2
        m = nn.Softmax(dim=2)
        dnn_out_test = m(dnn_out_test).detach().to('cpu').numpy()

        #prepare label and mask
        pad_mask_test, label_test = load_label(sp_pos_test, cf, le, 'cpu')
        pred = dnn_out_test.argmax(axis=2)
        #compute log likelihood for the batch samples
        res_batch = eval_log_llh(dnn_out_test, label_test, pad_mask_test)
        res_llh.append(np.array(res_batch))

        if bool(args.scanpath_gen_flag) == True:
            sn_len = (torch.max(torch.nan_to_num(word_ids_sn_test), dim=1)[0]+1-2).detach().to('cpu').numpy()
            #compute the scan path generated from the model when the first few fixed points are given
            sp_dnn = dnn.scanpath_generation(sn_emd=sn_input_ids_test,
                                                sn_mask=sn_attention_mask_test,
                                                word_ids_sn=word_ids_sn_test,
                                                sn_word_len = sn_word_len_test,
                                                le=le,
                                                max_pred_len=cf['max_pred_len'])

            sp_dnn, sp_human = prepare_scanpath(sp_dnn[0].detach().to('cpu').numpy(), sn_len, sp_pos_test, cf)
            sp_dnn_list.extend(sp_dnn)
            sp_human_list.extend(sp_human)
            text_list.extend(text_with_newlines)

        batch_indx +=1

res_llh = np.concatenate(res_llh).ravel()
loss_dict['test_ll'].append(res_llh)
loss_dict['fix_dur_mean'] = fix_dur_mean
loss_dict['fix_dur_std'] = fix_dur_std
loss_dict['landing_pos_mean'] = landing_pos_mean
loss_dict['landing_pos_std'] = landing_pos_std
loss_dict['sn_word_len_mean'] = sn_word_len_mean
loss_dict['sn_word_len_std'] = sn_word_len_std
print('\nTest likelihood is {} \n'.format(np.mean(res_llh)))

#save results
with open('{}/res_SAT_MECO_{}_eyettention_{}.pickle'.format(args.save_data_folder, args.test_mode, args.atten_type), 'wb') as handle:
    pickle.dump(loss_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

if bool(args.scanpath_gen_flag) == True:
    print("saving generated scanpath...")
    #save results
    dic = {"sp_dnn": sp_dnn_list, "sp_human": sp_human_list, "text": text_list}
    with open(os.path.join(args.save_data_folder, f'SAT_MECO_scanpath_generation_eyettention_{args.test_mode}_{args.atten_type}.pickle'), 'wb') as handle:
        pickle.dump(dic, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Finished!")
