import tokenizers
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing

from utils.config import config

"""
The k-mer tokenizer splits the DNA-seq into k-mers and applies tokenization and encoding based on a vocabulary
Note that 5-mer is the default configuration
"""
#initialize kmer tokenizer
kmer_tokenizer = tokenizers.Tokenizer(WordLevel("data/vocabs/" + str(config['kmer_size']) + "mer_vocab.json", unk_token="[UNK]"))
kmer_tokenizer.pre_tokenizer = Whitespace()
kmer_tokenizer.post_processor = TemplateProcessing(
    single="[CLS] $A [SEP]",
    pair="[CLS] $A [SEP] $B:1 [SEP]:1",
    special_tokens=[
        ("[PAD]", 0),
        ("[UNK]", 1),
        ("[CLS]", 2),
        ("[SEP]", 3),
        ("[MSK]", 4),
    ],
)
max_length = int(config["max_position_embeddings"]) - 2  #take additional "[CLS] $A [SEP]" into account
kmer_tokenizer.enable_truncation(max_length=max_length)
kmer_tokenizer.enable_padding(pad_id=0, pad_token="[PAD]", length=max_length, pad_to_multiple_of=config["attention_window"])