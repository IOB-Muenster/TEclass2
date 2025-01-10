from Bio import SeqIO
import pickle
import os

def save_dataset(dataset_train, dataset_valid, dataset_test, file_name):
    '''
        Saves a dataset to file_name with corresponding dashes (-train/-valid/-test)
    '''
    os.makedirs("/".join(file_name.split("/")[:-1]), exist_ok=True)
    with open(file_name+'-train.pkl', 'wb') as f:
        pickle.dump(dataset_train, f)
    with open(file_name+'-valid.pkl', 'wb') as f:
        pickle.dump(dataset_valid, f)
    with open(file_name+'-test.pkl', 'wb') as f:
        pickle.dump(dataset_test, f)

def load_dataset(file_name): 
    '''
        Loads and returns a dataset from a file_name with corresponding dashes (-train/-valid/-test)
    '''
    with open(file_name+'-train.pkl', 'rb') as f:
        dataset_train = pickle.load(f)
    with open(file_name+'-valid.pkl', 'rb') as f:
        dataset_valid = pickle.load(f)
    with open(file_name+'-test.pkl', 'rb') as f:
        dataset_test = pickle.load(f)
    return dataset_train, dataset_valid, dataset_test

def fasta2dict(datadict_, file_name):
    data = SeqIO.parse(file_name, "fasta")
    for l, entry in enumerate(iter(data)):
        try:
            description = entry.description
            if " " in description:
                seq_type = description.split(" ")[-1]
                if seq_type in datadict_.keys():
                    datadict_[seq_type].append([str(entry.seq), str(entry.id)])
        except Exception as e:
            raise Exception('Error while loading ', file_name, '\n', e)
    total_len = 0

    for key in datadict_.keys():
        key_len = len([d[0] for d in datadict_[key]])
        total_len += key_len
        print(key, "len: ", key_len) 
    print("total_len: ", total_len)
    return datadict_


def embl2dict(datadict_, file_name='data/Dfam_curatedonly.embl', file_type='embl'):
    '''
    Reads a file with annotated transposon sequences ad creates a dictonary with the following keywords
    which are gathered by the 'Type'-keyword in the annottaion for each sequence.
    This might differ for other filetypes and used databases. 
    Currently Dfam(_curatedonly).embl is officially supported and tested.
    '''
    data = SeqIO.parse(file_name, file_type)
   # data = SeqIO.index(file_name, file_type)
   # print(data)

    for l, entry in enumerate(iter(data)):
       
        try:
            seq_type = entry.annotations['comment']
            seq_subtype = seq_type[seq_type.find('SubType'):]
            seq_subtype = seq_subtype[9:seq_subtype.find('\n')]

            seq_type = seq_type[seq_type.find('Type'):] 
            seq_type = seq_type[6:seq_type.find('\n')]
            if seq_type in datadict_.keys():
                if seq_subtype in datadict_.keys():
                    datadict_[seq_subtype].append([str(entry.seq), str(entry.id)])
                else: 
                    datadict_[seq_type].append([str(entry.seq), str(entry.id)])
                continue
            #seq_type = entry.annotations['keywords']
            #if any([type for type in seq_type] in datadict_.keys()):
            #    datadict_[seq_type].append(str(entry.seq))
            #    continue
            else:
                #print(seq_type, 'discarded') 
                #Unknown, Retroposon, RC, Sattelite
                continue

        except Exception as e:
            raise Exception('Error while loading ', file_name, '\n', e)

    total_len = 0

    for key in datadict_.keys():
        key_len = len([d[0] for d in datadict_[key]])
        total_len += key_len
        print(key, "len: ", key_len) 
    print("total_len: ", total_len)
    return datadict_


def load_classification_file(file_name, file_type='fasta'):
    '''
        Loads a file which can be used for classification
        Currently only fasta/fastq tested 
    '''
    data = SeqIO.parse(file_name, file_type)
    data_ = []
    for l, entry in enumerate(iter(data)):
        try:
            entry_ = [str(entry.seq), str(entry.id)]
            data_.append(entry_)
            
        except Exception as e:
            raise Exception('Error while loading ', file_name, '\n', e)

    return data_


def load_vocab(file_name='data/5mer_vocab'):
    '''
    Loads a specified vocab txt file dictionary for the transformer
    '''
    vocab_file = {}
    with open(file_name+'.txt', 'r', encoding="utf-8") as f:
        tokens = f.readlines()
    for i, token in enumerate(tokens):
        token = token.rstrip("\n")
        vocab_file[token] = i


    return vocab_file

#
#def write_vocab_json(file_name='data/new_vocab'):  
#    '''
#    Writes a dict to a utf-8 json file
#    '''
#    import json
#    vocab_file = load_vocab(file_name)
#    with open(file_name+'.json'.format(1), 'w', encoding='utf-8') as file:
#        json.dump(vocab_file, file, ensure_ascii=False)
    

def create_vocab(k=5):
    """
    Creates a kmer-vocabulary
    Needs manual post-processing to remove all ' which are added to python strings
    """
    import itertools
    list = ["".join(x) for x in itertools.product(["A","G", "T", "C", "N"],repeat=k)]

    dict_list = []
    dictionary = {}
    dictionary["[PAD]"] = 0
    dictionary["[UNK]"] = 1
    dictionary["[CLS]"] = 2
    dictionary["[SEP]"] = 3
    dictionary["[MSK]"] = 4
    for i, k in enumerate(list):
        dictionary[k]= i+5

    print(dictionary)
    exit()
