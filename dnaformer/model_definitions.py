import torch
import torch.nn as nn
from transformers import Trainer, LongformerConfig, LongformerForSequenceClassification

from utils.config import config
import utils.io_handler as io
from utils.dataset_utils import kmer2seq
from math import pi

vocab_file = io.load_vocab('data/vocabs/' + str(config['kmer_size']) + 'mer_vocab')
global model 
global model_config 


def compute_kernel(x, y):
    '''
        Computes the kernels for the loss function
    '''
    # MMD Kernel
    x_size = x.size(0)
    y_size = y.size(0)
    dim = x.size(1)
    x = x.unsqueeze(1) # (x_size, 1, dim)
    y = y.unsqueeze(0) # (1, y_size, dim)
    tiled_x = x.expand(x_size, y_size, dim)
    tiled_y = y.expand(x_size, y_size, dim)
    kernel_input = (tiled_x - tiled_y).pow(2).mean(2)/float(dim)
    return torch.exp(-kernel_input) # (x_size, y_size)
    
    # evidence lower bound (ELBO) criterion
    x = x.view(x.size(0), x.size(1)) # * x.size(2))
    y = y.view(y.size(0), y.size(1)) # * y.size(2))

    xx, yy, xy = torch.mm(x,x.t()), torch.mm(y,y.t()), torch.mm(x,y.t())

    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))

    return xx, yy, xy, rx, ry

    
def compute_mmd(x, y):
    '''
        Computes the MMD-Loss between x and ground-truth Distribution y
    '''

    x_kernel = compute_kernel(x, x)
    y_kernel = compute_kernel(y, y)
    xy_kernel = compute_kernel(x, y)
    mmd = x_kernel.mean() + y_kernel.mean() - 2*xy_kernel.mean()
    return mmd

    xx, yy, xy, rx, ry = compute_kernel(x, y)
    batch_size = x.size(0)

    K = torch.exp(- (rx.t() + rx - 2*xx))
    L = torch.exp(- (ry.t() + ry - 2*yy))
    P = torch.exp(- (rx.t() + ry - 2*xy))

    beta = (1./(batch_size*(batch_size-1)))
    gamma = (2./(batch_size*batch_size)) 

    mmd = beta * (torch.sum(K)+torch.sum(L)) - gamma * torch.sum(P)
    return mmd


def get_model():
    '''
        Sets the global selected model type with their given configuration
    '''
    global model, model_config

    model_config = LongformerConfig(attention_window = config["attention_window"], 
                                vocab_size = len(vocab_file),
                                max_position_embeddings = config["max_position_embeddings"], 
                                num_labels = len(config["te_keywords"]),
                                #num_labels=config["num_labels"],
                                hidden_size = config["hidden_size"],
                                num_hidden_layers = config["num_hidden_layers"],
                                num_attention_heads = config["num_attention_heads"],
                                intermediate_size = config["intermediate_size"],
                                position_embedding_type = config["position_embedding_type"],
                                problem_type = "single_label_classification",
                                output_attentions= (not config["train"] and config["classification"] and not config["low_memory"]),
                                return_dict=True,
                                pad_token_id = 0,
                                bos_token_id = 2,
                                eos_token_id = 3)
    model = LongformerForSequenceClassification(model_config)


    return model

def get_model_config():
    '''
        Helper function, returns current config of the specified model
    '''
    return model_config



class DNAFormer_Trainer(Trainer):
    '''
        custom trainer which overrides compute_loss with WCE and supports VAE training with additional mmd-loss
    '''
    def __init__(self, sample_weight=[], *args, **kwargs):

        if not config['classification']:
                self.sample_weight = sample_weight
                self.loss_fct = torch.nn.CrossEntropyLoss(weight=self.sample_weight)
            
        super(DNAFormer_Trainer, self).__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        #if not self.vae:
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # compute custom loss
        loss = self.loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))

        
        return (loss, outputs) if return_outputs else loss


class Distributions():
    '''
        Distirbution class which returns defined distributions
        Gauss - no restraints
        circle & cross - only with latend_dim = 2
    '''
    def __init__(self, dist_type, nsamples, dim):
        self.type = dist_type
        self.nsamples = nsamples * 5
        self.dim = dim

        self.radius = 10                    #for circle
        self.expansion = 1.5                #for circle
        self.angle = torch.tensor(2 * pi)   #for cross

    def __call__(self):
        if self.type == 'gauss':
            dist = torch.randn(self.nsamples, self.dim, device=config['device'])
        elif self.type == 'circular':
            # random angle and radius
            angle = 2 * pi * torch.randn(self.nsamples, self.dim, device=config['device'])
            r = self.radius + torch.randn(self.nsamples, self.dim, device=config['device'])*self.expansion

            # coordinates
            x = r * torch.cos(angle)
            y = r * torch.sin(angle)
            dist = torch.concat((x,y), axis=1).reshape(self.nsamples*2, self.dim)
            
        elif self.type == 'cross':
            x = torch.randn(self.nsamples, self.ndim, device=config['device'])
            y = torch.randn(self.nsamples, self.ndim, device=config['device'])

            x = (x * torch.cos(self.angle) * y)
            y = (y * torch.sin(self.angle) * x)
            dist = torch.concat((x,y), axis=1).reshape(self.nsamples*2, self.dim)

        else: raise Exception("No such distribution: ", self.type)
        return dist
