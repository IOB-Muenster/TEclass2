from utils.config import config
from utils.augmentation import normalize
import utils.io_handler as io

#create empty dict
te_keywords = config["te_keywords"]
datadict_ = {i:[] for i in te_keywords}
classification_map = {i:te_keywords.index(i) for i in te_keywords}
classification_map_int = {int(str(classification_map[k])):k for k in classification_map}

def seq2kmer(seq, embedd_larger_seq=True, max_len = config["max_position_embeddings"], k=config["kmer_size"]):
    '''
        Creates a whitespace splitted list of kmers
    '''
    #dilate kmers with larger embedding_window 'w'
    if embedd_larger_seq:
        w = min(max(1, int(len(seq) / max_len)), k-1) #compute w dynamically up to k steps
    else: w=1
    kmer = [seq[i:i+k] for i in range(0, len(seq)+1-k, w)]
    kmer = " ".join(kmer)
    return kmer, w


def kmer2seq(kmers, w):
    '''
        Creates the original (eventually cropped) sequence from kmers
        Note that depending on kmer size the return list is k-1-bases shorter,
        except the sequence is larger than the max_embedding_size
    '''
    seq = "".join([kmer[:w] for kmer in kmers])
    return seq


def split_dataset(dataset, train=0.75, valid=0.15, test=0.1):
    '''
        Splits the dataset-dictionary into several lists based upon train/valid/test
    '''
    train_len = int(train*len(dataset))
    valid_len = int(valid*len(dataset))
    return dataset[:train_len], dataset[train_len:train_len+valid_len], dataset[train_len+valid_len:]


def dict2dataset(data_, save=True, save_file_name=config["dataset_path"], normalization=True, split=True):
    '''
        Returns a splitted preprocessed dataset
    '''
    #change to list style
    dataset_train, dataset_valid, dataset_test = [], [], []
    norm = normalize()
    for key in data_.keys():
        dataset_ = []
        for seq in (j for j in data_[key]):
            if normalization: 
                dataset_ += [[norm(seq[0]), seq[1], classification_map[key]]]
            else: dataset_ += [[seq[0], seq[1], classification_map[key]]]

        if split:
            train, valid, test = split_dataset(dataset_)
            dataset_train += list(train)
            dataset_valid += list(valid)
            dataset_test  += list(test)
        else:
            dataset_test += list(dataset_)

    if split and save: io.save_dataset(dataset_train, dataset_valid, dataset_test, save_file_name)
    
    if split: return dataset_train, dataset_valid, dataset_test
    else: return dataset_test



def create_new_dataset(file_name, split=True, save=False):
    '''
        Create a new dataset based upon sequence files
    '''
    if config['classification'] and not config['prediction']:
        return io.load_classification_file(file_name)
    elif not config['classification'] and config['prediction']:
        dict = None
        if file_name.endswith(".embl"):
            dict = io.embl2dict(datadict_, file_name=file_name)
        elif file_name.endswith(".fa") or file_name.endswith(".fasta"):
            dict = io.fasta2dict(datadict_, file_name=file_name)
        print('Dataset created')
        return dict2dataset(dict, save=save, split=split)
    else:
        raise Exception("Please just choose classification or prediction!")



def save_histogram(dataset):
    '''
        Creates a length distribution histogram based upon a Transposondataset
    '''
    import matplotlib.pyplot as plt
    import numpy as np

    list = []
    max = 0
    ignored = 0
    over_512 = 0
    range_max = 3000

    for key in dataset.keys():
        for i in range(len(dataset[key])):
        
            dataset[key][i] = len(dataset[key][i][0])
            if max < dataset[key][i]: max = dataset[key][i]
            if dataset[key][i] > range_max: ignored += 1
            if dataset[key][i] > 512: over_512 +=1

        
        list += [dataset[key]]
    
    print(max, ignored, (20000-ignored)/20000, over_512/20000)
    plt.hist(list, stacked=True, range=(0, range_max), bins=200, )
    plt.legend(datadict_)
    plt.title('Transposon length distribution up to ' + str(range_max) +' bases (approx. X% of sequences)')
    plt.tight_layout()
    plt.show()
