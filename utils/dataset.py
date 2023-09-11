from torch.utils.data import Dataset
import torch
import random
import numpy as np

from utils.augmentation import transform
import dnaformer.tokenizer as tokenizer
from utils.dataset_utils import *
from utils.config import config

tokenizer = tokenizer.kmer_tokenizer


class TransposonDataset(Dataset):
    '''
        Pytorch Dataset for handling the created written transposon dataset
        embedd_larger_seq: If True use dilated kmers which reduces the sequence impact up to w times
        train: Wheter to apply augmentation
    '''
    def __init__(self, data, tokenizer, embedd_larger_seq=True, train=False):
        #save original sequences
        self.seqs = [seq[0] for seq in data]

        #save sequence ids
        self.ids = [id[1] for id in data]

        #save kmer embeddings and embedding_with when kmers are dilated
        embeddings = [seq2kmer(seq[0], embedd_larger_seq) for seq in data]
        kmers, embed_w =  [kmer[0] for kmer in embeddings], [kmer[1] for kmer in embeddings]
        self.embed_w = embed_w

        #tokenize kmer into ids and attention masks
        #tokenized_input = [tokenizer.encode(kmer) for kmer in kmers]
        #self.encoded_kmers = [enc_ids.ids for enc_ids in tokenized_input]
        #self.attention_masks = [att_mask.attention_mask for att_mask in tokenized_input]

        self.global_att_tokens = np.array(config["global_att_tokens"])
        self.beta = config["wce_scaling"]

        #save labels
        self.labels = [label[2] for label in data]

        #save additional parameters
        self.tokenizer = tokenizer
        self.train = train

        sample_weight = []
        eps = 0
        for i in range(len(datadict_.keys())): 
            sample_weight.append(self.labels.count(i)+eps)
        self.sample_weight = [1 / (x / sum(sample_weight)) for x in sample_weight] #get inverse of occurence weigths -> large occurences weigth less
        #self.sample_weight = [x / sum(self.sample_weight) for x in self.sample_weight] #additional normalization to 0-1
        self.sample_weight = torch.tensor(self.sample_weight, device = config['device'])

        uniform = torch.tensor(1/len(datadict_.keys()), device = config['device'])
        uniform = uniform.repeat(len(datadict_.keys()))
        self.sample_weight = self.beta * self.sample_weight + (1-self.beta) * uniform
        

        print("dataset size: " , len(self.labels))

    def getembedding_w(self, idx):
        embed_w = self.embed_w[idx]
        return embed_w

    def getoriginalseq(self, idx):
        return self.seqs[idx], self.ids[idx]
    
    def getseqids(self):
        return self.ids

    def getkmer(self, seq_index, kmer_pos):
        seq = self.seqs[seq_index]

        if kmer_pos >= len(seq2kmer(seq)[0].split()):
            return '[PAD]'   #attention on padding

        kmer = seq2kmer(seq)[0].split()[kmer_pos]
        return kmer

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        seq = self.seqs[idx]

        #kmers = self.encoded_kmers[idx]
        #att_mask = self.attention_masks[idx]

        #Apply augmentation 
        if self.train and random.random() > 1-config["augmentation_probability"]:
            seq = transform(seq)

        kmers = seq2kmer(seq)[0]
        encoded_input = self.tokenizer.encode(kmers)
        kmers = encoded_input.ids
        att_mask = encoded_input.attention_mask

        #generate global attention mask
        global_att_mask = torch.zeros_like(torch.tensor(att_mask), dtype=torch.long) #, device=att_mask.device)
        global_att_tokens = self.global_att_tokens[self.global_att_tokens<=len(kmers)]   
        global_att_mask[global_att_tokens] = 1  #global attentions
        
        return {"input_ids": kmers, "attention_mask": att_mask, "global_attention_mask": global_att_mask, "label" : label}   

class ClassificationDataset(Dataset):
    '''
        Pytorch Dataset for handling the unknwon to be classified data
        embedd_larger_seq: If True use dilated kmers which reduces the sequence impact up to w times
    '''
    def __init__(self, data, tokenizer, embedd_larger_seq=True):
        #save original sequences
        self.seqs = [seq[0].upper() for seq in data]
        #save sequence ids
        self.ids = [id[1] for id in data]

        #save kmer embeddings and embedding_with when kmers are dilated
        embeddings = [seq2kmer(seq[0].upper(), embedd_larger_seq) for seq in data]
        kmers, embed_w =  [kmer[0] for kmer in embeddings], [kmer[1] for kmer in embeddings]
        self.embed_w = embed_w

        tokenized_input = [tokenizer.encode(kmer) for kmer in kmers]
        self.encoded_kmers = [enc_ids.ids for enc_ids in tokenized_input]
        self.attention_masks = [att_mask.attention_mask for att_mask in tokenized_input]

        self.global_att_tokens = np.array(config["global_att_tokens"])

        print("classification dataset size: " , len(self.ids))

    def getembedding_w(self, idx):
        embed_w = self.embed_w[idx]
        return embed_w

    def getoriginalseq(self, idx):
        return self.seqs[idx], self.ids[idx]
    
    def getseqids(self):
        return self.ids

    def getkmer(self, seq_index, kmer_pos):
        seq = self.seqs[seq_index]

        if kmer_pos >= len(seq2kmer(seq)[0].split()):
            return '[PAD]'   #attention on padding

        kmer = seq2kmer(seq)[0].split()[kmer_pos]
        return kmer

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        att_mask = self.attention_masks[idx]
        kmers = self.encoded_kmers[idx]
        #generate global attention mask
        global_att_mask = torch.zeros_like(torch.tensor(att_mask), dtype=torch.long) #, device=att_mask.device)
        global_att_tokens = self.global_att_tokens[self.global_att_tokens<len(kmers)]   
        global_att_mask[global_att_tokens] = 1  #global attentions
        return {"input_ids": kmers, "attention_mask": att_mask, "global_attention_mask": global_att_mask}    


def return_datasets(tokenizer, file_name, datasets=None, split=True):
    '''
        Loads the datasets as Dataset-object and applies the specified tokenizer to all inputs
    '''
    if split:
        if datasets == None:
            #Dataset is not provided, so load default
            #This is included for faster processing for train/test dataset
            dataset_train, dataset_valid, dataset_test = io.load_dataset(file_name)
        else: 
            #Dataset is provided as it is newly created and given as parameter
            dataset_train, dataset_valid, dataset_test = datasets

        dataset_train = TransposonDataset(dataset_train, tokenizer, train=True) #apply with augmentation
        dataset_valid = TransposonDataset(dataset_valid, tokenizer)
        dataset_test = TransposonDataset(dataset_test, tokenizer)
        return dataset_train, dataset_valid, dataset_test
    else:
        #create either a transposondataset for training or a classificationdataset for classification
        if config['prediction']:
            #in this case we read from a fasta file
            dataset_predict = TransposonDataset(datasets, tokenizer)
            return dataset_predict
        elif config['classification']:
            dataset_classify = ClassificationDataset(datasets, tokenizer)
            return dataset_classify
        else:
            raise Exception("Only select prediction or classification.")
