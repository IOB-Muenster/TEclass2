import random
import numpy as np
from torchvision import transforms

'''
This file holds different String augmentation which are biological event-inspired and are based upon DNA-sequence statistics.
Supported types are Insertion, Deletion, Single Nucleotide Polymorphisms (SNPs), Ambiguity/Masking (AGTC -> N), Transposon injecting, Reverse and Complement
The input has to of AGTCN
'''

#holds the likeability of specific SNPs for each AGTC/AGTC mapping
snp_matrix = [[0,1,1,1],    #A
              [1,0,1,1],    #G
              [1,1,0,1],    #T
              [1,1,1,0],    #C
              [1,1,1,1]]    #ambuigity N will be mapped to AGTC

#define the mappings as literal dor each base
base2key_map = {'A':0 , 'G': 1, 'T':2, 'C':3, 'N':4}
key2base_map = {0:'A' , 1:'G', 2:'T', 3:'C', 'N':4}

#define the complement for each base
complement_map = {ord('G'):'C', ord('T'):'A', ord('C'):'G', ord('A'):'T', ord('N'):'N'}
complement_map_int = {ord('0'):'A', ord('1'):'G', ord('2'):'T', ord('3'):'C', ord('4'):'N'}




def one_of(transform_list):
    return transform_list[random.randint(0, (len(transform_list)-1))]


class normalize(object):
    '''
        Uppercases all characters and removes all bases not in base2key_map to 'N'
    '''
    def __call__(self, seq):
        seq = seq.upper()
        if not all(s in base2key_map for s in seq):
            replace_chars = [char for char in seq if char not in base2key_map.keys()]
            replace_chars = list(dict.fromkeys(replace_chars))
            for c in replace_chars:
                seq = seq.replace(c, 'N')
        return seq

class snp(object):
    '''
        creates a number of n SNPs based upon the likeability matrix on the given seq
        Does NOT introucde ambuigity (AGTCN -> AGTC)
    '''
    def __init__(self, n=1, matrix=snp_matrix):
        self.n = n
        self.snp_matrix = matrix

    def __call__(self, seq):
        for i in range(self.n):
            j = random.randint(0, len(seq)-1)
            snp_ = base2key_map[seq[j]]
            snp_ = self.snp_matrix[snp_] * np.random.rand(4)
            snp_ = key2base_map[int(snp_.max())]
            seq = seq[:j] + snp_ + seq[j+1:]
        return seq

class mask(object):
    '''
        Masks n times a sequence of 'length' concurrent characters into ambuigity character N (AGTCN -> N)
        Note that this mask is NOT equal to masking with ['MSK']-token during tokenization. They have different purposes.
    '''
    def __init__(self, n=1, length=5, pos=[0.05,0.95]):
        self.n = n
        self.length = length
        self.pos = pos

    def __call__(self, seq):
        for i in range(self.n):
            j = int(random.uniform(*self.pos) * len(seq))
            mask_ = 'N'*self.length
            seq = seq[:j] + mask_ + seq[j+self.length:]
        return seq

class insertion(object):
    '''
        Inserts a random sequence (insert_seq=None) of length in [min_lentgh, max_length] at a random position.
        A specified sequence can also be inserted; the length is then omitted
    '''
    def __init__(self, min_length=5, max_length=20, pos=[0.05,0.95], insert_seq=None):
        self.min_length = min_length
        self.max_length = max_length
        self.insert_seq = insert_seq
        self.pos = pos

    def __call__(self, seq):
        j = int(random.uniform(*self.pos) * len(seq))
        if self.insert_seq: insert_ = self.insert_seq
        else:
            ##create random sequence
            length = random.randint(self.min_length, self.max_length)
            rand_list = random.choices(range(0, 3), k=length)
            rand_list = "".join(str(e) for e in rand_list)
            insert_ = rand_list.translate(complement_map_int)
        return seq[:j] + insert_ + seq[j:]

class deletion(object):
    '''
        Deletes n times a subsequence of 'length' concurrent characters
    '''
    def __init__(self, n=1, min_length=5, max_length=20, pos=[0.05,0.95]):
        self.n = n
        self.min_length = min_length
        self.max_length = max_length
        self.pos = pos
        
    def __call__(self, seq):
        for i in range(self.n):
            j = int(random.uniform(*self.pos) * len(seq))
            length = random.randint(self.min_length, self.max_length)
            seq = seq[:j] + seq[j+length:]
        return seq

class repeat(object):
    '''
        Forward repeats a sequence-part of 'length' with a distance of 'min_distance' characters
        The insert position is between 'pos'% of the sequence
        Should be called at last step
    '''
    def __init__(self, length=5, min_dist=0, pos=[0.05,0.95]):
        self.min_dist = min_dist
        self.length = length
        self.pos = pos


    def __call__(self, seq):
        min_dist = (self.min_dist / len(seq))    #to 0.0-1.0 map
        j = int((len(seq)-self.length) * random.uniform(self.pos[0], 1-self.min_dist))
        repeat_ = seq[j:j+self.length]  #get part seq
        j = int(len(seq) * random.uniform((j+self.length)/len(seq)+self.min_dist, self.pos[1]))
        seq = seq[:j] + repeat_ + seq[j:]
        return seq



class reverse(object):
    '''
        Returns the reverse of the sequence
    '''
    def __call__(self, seq):
        return seq[::-1]

class complement(object):
    '''
        Returns the complement of a DNA sequence
        G<->C, T<->A, N<->N
    '''
    def __init__(self, complement_map=complement_map):
        self.complement_map = complement_map

    def __call__(self, seq):
        return seq.translate(complement_map)
  
class reverse_complement(object):
    '''
        Returns the reverse complement of a DNA sequence
    '''
    def __init__(self, complement_map=complement_map):
        self.complement_map = complement_map
    def __call__(self, seq):
        return seq[::-1].translate(complement_map)
    
    
class add_tail(object):
    '''
        Adds a tail to the seq with a given random length of tail_type
    '''
    def __init__(self, tail_type='A', length=[5,20]):
        self.tail_type = tail_type
        self.length = length

    def __call__(self, seq):
        j = random.randint(*self.length)
        return seq + self.tail_type*j

class remove_tail(object):
    '''
        Removes a tail of a seq of tail_type
    '''
    def __init__(self, tail_type='A'):
        self.tail_type = tail_type

    def __call__(self, seq):
        while (seq[::-1].find(self.tail_type))<=0:
            seq = seq[:-2]
        return seq


class inject_transposons(object):
    '''
    Inject a random transposon at a random position into the sequence. Can also create tandem site duplications (TSD).
    '''
    def __init__(self, pos=[0.05,0.95], create_tsd=True, tsd_len=[5,20]):
        self.pos = pos
        self.create_tsd = create_tsd
        self.tsd_len = tsd_len

    def __call__(self, seq):
        transposon = ''    #get from database
        if self.create_tsd:
            tsd = ''    #create tsd
            transposon = tsd + transposon + tsd
        j = int((len(seq)-len(transposon)) * random.uniform(*self.pos))
        seq = seq[:j] + transposon + seq[j:]

        return seq

class identity(object):
    '''
        Returns the identity (changes nothing)
    '''
    def __call__(self, seq):
        return seq


class compose_(object):
    def __init__(self, list):
        self.list = list
    def __call__(self, seq):
        return one_of([*self.list])(seq)

#define a composed transformation list which are used for each sequence
transform = transforms.Compose([normalize(),
                                mask(),
                                compose_([
                                    insertion(max_length=7),
                                    identity()
                                ]),
                                compose_([
                                    deletion(n=1, max_length=7),
                                    identity()
                                ]),
                                compose_([
                                    repeat(),   
                                    identity()
                                ]),                       
                                compose_([
                                    complement(), 
                                    reverse(), 
                                    reverse_complement(),
                                    identity()
                                ]),
                                compose_([
                                    add_tail(), 
                                    remove_tail(), 
                                    identity()
                                ])
])