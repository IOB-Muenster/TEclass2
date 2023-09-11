from datetime import datetime
from utils.dataset import classification_map_int, datadict_
import matplotlib.pyplot as plt
import numpy as np
from utils.config import config
from tqdm import tqdm
from scipy import interpolate
from numpy.lib.stride_tricks import as_strided
from sklearn.metrics import  ConfusionMatrixDisplay
import pandas as pd
from torch.nn import Softmax
import torch
import os

kernel_size = 20
kernel = np.ones(kernel_size) / kernel_size

def save_confusion_matrix(labels, preds):
    '''
        Save a confusion matrix based on given label and predictions
    '''
    #preds = [str(p).translate(classification_map_int) for p in preds]
    #labels = [str(l).translate(classification_map_int) for l in labels]
    preds = [classification_map_int[int(p)] for p in preds ]
    labels = [classification_map_int[int(l)] for l in labels ]
    fig = ConfusionMatrixDisplay.from_predictions(y_true=labels, y_pred=preds)
    fig.plot()
    plt.rcParams['font.size'] = 6
    plt.savefig(config['model_save_path'] + config['model_name'] + '/confusion_matrix.pdf', bbox_inches='tight')
    plt.clf()
    plt.close()
    return

def visualize_and_save(dataset, output_logits_, local_attentions_, global_attentions_, save_imgs):
    """
        Visualizes local and global attention as heatplots and saves them
    """
    #LAYER sequenzen ATTENTION_HEADS_PER_LAYER each_seq_token_of_selected_seq attentions_vals
    lc_att_sum = sum(local_attentions_[:])
    gl_att_sum = sum(global_attentions_[:])

    save_path_ = config['vis_save_path']
    inf_arr = []    #save gathered information for table output

    for index in tqdm(range(len(output_logits_))):
        predicted = str(output_logits_[index].argmax(-1)).translate(classification_map_int) # + " (" + str(label_ids_[index]).translate(classification_map_int) + ")"
        seq_len = len(dataset.getoriginalseq(index)[0])
        seq_id = dataset.getoriginalseq(index)[1]
        embed_w = dataset.getembedding_w(index)

        #create subdirectories
        save_path = os.path.join(save_path_, seq_id)
        try: 
            os.makedirs(save_path)
        except OSError: 
            pass
            #print('Folder already exists')  
        save_path = save_path + '/'

        fig_1, lc_att = local_attention_plot(lc_att_sum[index], seq_len, embed_w)
        if save_imgs: save_vis(fig_1, save_path, 'local_att', seq_id, predicted, ticks=True)

        fig_2, flt_local_att = flatted_local_attention_plot(lc_att, seq_len, embed_w)
        if save_imgs: save_vis(fig_2, save_path, 'local_flat_att', seq_id,  predicted)

        fig_3, glb_att = global_attention_plot(gl_att_sum[index], seq_len, embed_w)
        if save_imgs: save_vis(fig_3, save_path, 'global_att', seq_id, predicted)

        fig_4 = sequence_attention_plot(flt_local_att, glb_att)
        if save_imgs: save_vis(fig_4, save_path, 'seq_att', seq_id, predicted, ticks=True)


        #if not save_imgs:
        plt.close(fig_1), plt.close(fig_2), plt.close(fig_3), plt.close(fig_4)

        top_kmers = get_top_kmers(dataset, index, glb_att, 10)
        motif_pos, motif_seq = get_motif_seq(dataset, index, flt_local_att)

        inf_arr.append([seq_id, top_kmers, motif_pos, motif_seq])
    generate_tsv(output_logits_, inf_arr)
    return


def save_vis(fig, path, type, seq_id, predicted, ticks=False):
    '''
        Top level Convenience function to save a pdf plot
    '''
    #plt.title(str(seq_id))
    plt.xlabel(str(seq_id) + ' predicted: ' + predicted)
    if not ticks:
        plt.tick_params(labelleft=False, left=False)
    plt.tight_layout()
    plt.savefig(path + str(type) + '.pdf', bbox_inches='tight')
    fig.clf()
    plt.close(fig)
    return 


def generate_tsv(predictions, inf_arr, dataset=None):
    '''
        Generate tabular output
    '''
    #create subdirectories
    save_path = config['vis_save_path']
    try: 
        pass
        #os.mkdir(config['vis_save_path'])
    except OSError: 
        pass
        #print('Folder already exists')  

    preds = predictions.argmax(-1)
    preds = [str(p).translate(classification_map_int) for p in preds]

    softmax = Softmax(dim=0)
    confidences =  [softmax(p).numpy() for p in torch.tensor(predictions)]

    confidences = np.array(list(zip(*confidences)))

    if not config["low_memory"]:
        top_k_mers = [kmer[1] for kmer in inf_arr]
        motif_pos = [motif[2] for motif in inf_arr]
        motif_seq = [motif[3] for motif in inf_arr]

        seqids = [id[0] for id in inf_arr]
    else:
        seqids = [dataset.getoriginalseq(index)[1] for index in range(0, len(preds))]
        dataframe = {
                    'seq-id': seqids,
                    'DNA': confidences[3]#,
                    }

    date_now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    outfile_name = config["vis_save_path"] + '/' + str(date_now) + "_predictions.tsv" 
    outfile_handler = open(config['vis_save_path'], 'w')

    dict_sort = dict(sorted(classification_map_int.items()))
    head = "\t".join(["name", "class", "max_probability"] + list(dict_sort.values()))
    n_seqs = len(confidences[0]) 
    n_cats = len(confidences) 

    with outfile_handler as fh:
        fh.write(head + "\n")
        for i in range(0, n_seqs):
            te_class = classification_map_int[int(preds[i])]
            name = seqids[i] + "\t"
            values = []
            scores = ""
            for j in range(0, n_cats):
                scores += "\t" + '{:.3f}'.format(confidences[j][i])
                values.append(confidences[j][i])
            linefull= name + te_class + "\t" + '{:.3f}'.format(max(values)) + scores 
            fh.write(linefull + "\n")
    outfile_handler.close()


def expand_and_interpolate(attentions_, embedding_w):
    '''
        returns an interpolated expanded array depending on the kmer embedding width
    '''
    #interpolate to original sequence length
    if embedding_w > 1:
        max_seq_len = config["max_position_embeddings"]*embedding_w

        if attentions_.ndim == 1:
            attentions_expanded = np.arange(max_seq_len)
            attentions_ = attentions_[:max_seq_len]

            f = interpolate.interp1d(np.arange(0, len(attentions_)), attentions_)
            attentions_expanded = f(np.linspace(0.0, len(attentions_)-1, len(attentions_expanded)))
            #attentions_expanded = (attentions_expanded - np.min(attentions_expanded))/np.ptp(attentions_expanded)
        else:
            #2d for local_att
            r, c = attentions_.shape                                    # number of rows/columns
            rs, cs = attentions_.strides                                # row/column strides 
            x = as_strided(attentions_, (r, embedding_w, c, embedding_w), (rs, 0, cs, 0)) # view a as larger 4D array

            return x.reshape(r*embedding_w, c*embedding_w) 

        return attentions_expanded

    #attentions_ = (attentions_ - np.min(attentions_))/np.ptp(attentions_)
    return attentions_


def flatted_local_attention_plot(local_attentions_, seq_len, embed_w):
    '''
        Computes the local attention as a flatted/reduced 1D heatplot
    '''
    att_window = int(config["attention_window"]/2)
    w = min(seq_len, config["max_position_embeddings"]*embed_w) #get shortest possible embedding

    #compute sum over all heads and all layers
    flatted_local_att = np.sum(local_attentions_[:w], axis=1)

    #ignore CLS and SEP tokens due to no attention
    flatted_local_att = (flatted_local_att[1:-2] - np.min(flatted_local_att[1:-2]) ) / (np.max(flatted_local_att[1:-2]) - np.min(flatted_local_att[1:-2]))
    #flatted_local_att = expand_and_interpolate(flatted_local_att, embed_w)
    flatted_local_att = np.convolve(flatted_local_att, kernel, mode='same')
    flatted_local_att = (flatted_local_att - np.min(flatted_local_att)) / np.ptp(flatted_local_att)

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    im = ax1.imshow(flatted_local_att.reshape(1,len(flatted_local_att)), cmap='hot', interpolation='none')
    ax1.set_aspect(5*(seq_len/config["max_position_embeddings"]) + 5)
    fig.colorbar(im, orientation = 'horizontal')

    return fig, flatted_local_att
    


def global_attention_plot(global_attentions_, seq_len, embed_w):
    """
        Computes global attention heatplots as sum over all layers and heads
    """
    w = min(seq_len, config["max_position_embeddings"]*embed_w)
    global_attentions_ = np.squeeze(np.sum(sum(global_attentions_), -1))[:w] #.reshape(config["max_position_embeddings"]*embed_w)
    global_attentions_ = expand_and_interpolate(global_attentions_, embed_w)
    #if embed_w > 1:
    #    w =  config["max_position_embeddings"]*embed_w
    global_attentions_ = global_attentions_.reshape(1, w) /np.ptp(global_attentions_)

    #plot heatmap
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    im = ax1.imshow(global_attentions_, cmap='hot', interpolation='none')
    ax1.set_aspect(5*(seq_len/config["max_position_embeddings"]) + 5)
    ax1.figure.colorbar(im, orientation = 'horizontal')

    return fig, global_attentions_.squeeze()


def local_attention_plot(local_attentions_, seq_len, embed_w):
    '''
        Computes local attention heatplots as sum over all layers
    '''
    w = min(seq_len, config["max_position_embeddings"]) #get shortest possible embedding
    att_window = int(config["attention_window"]/2)

    #compute sum over all heads and normalize
    local_attentions_ = sum(local_attentions_)
    
    #local_attentions_[1:-2] = (local_attentions_[1:-2] - np.min(local_attentions_[1:-2])) #/np.ptp(local_attentions_[1:-2])
    local_attentions_ = (local_attentions_) / np.ptp(local_attentions_)

    #array of w*att_window into w*w
    exp_att = np.arange(w * w, dtype=float)
    exp_att = exp_att.reshape((w , w))
    exp_att[:,:] = 0
    #create local attention heatmap of output
    for l, token in enumerate(local_attentions_[:w]):
        start = max(0, l-att_window)
        end = min(w-1, l+att_window)
 
        att_window_start = max(0, att_window - l)
        att_window_end = min(att_window*2,  att_window_start + end-start)
        att_scores = token[att_window_start : att_window_end]
        exp_att[start : end , l] = att_scores

    exp_att = expand_and_interpolate(exp_att, embed_w)
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    im = ax1.imshow(exp_att, cmap='hot', interpolation='none')
    ax1.figure.colorbar(im, ax=ax1)
    return fig, exp_att

def sequence_attention_plot(local_attention_, global_attention_):
    """
        Computes the attention globally and locally as a plot instead of heatmap
    """
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(range(len(global_attention_)), global_attention_, label="global")
    ax1.plot(range(len(local_attention_)), local_attention_, label="local")
    #ax1.set_aspect(100)
    ax1.legend()

    return fig


def get_top_kmers(dataset, index, attention, top_n):
    '''
        Returns top n k-mers based on attention array
    '''
    top_kmers_pos = np.argpartition(attention, -top_n)
    top_kmers = [dataset.getkmer(index, pos) for pos in top_kmers_pos[-top_n:]]

    return top_kmers


def get_motif_seq(dataset, index, attention):
    '''
        Returns the longest highest scoring motif sequence over given threshold
    '''
    min_len = 7
    threshold = np.average(attention)
    attention = attention > threshold   #convert to bool
    arr_chg = np.where(attention[:-1] != attention[1:])[0]  #log all occuring changes

    if arr_chg.shape != (2,):   #we have multiple ones
        splits = np.split(arr_chg, len(arr_chg)/2)

        #split arrays and get longest split where no change occurs
        max_len = [len[1]-len[0] for len in splits]
        arr_chg = splits[np.argmax(max_len)]

        start = arr_chg[0]+1
        end = arr_chg[1]
        if attention[start] == True and (end-start) >= min_len:
            seq = dataset.getoriginalseq(index)[0][start:end]
        else:
            return "", ""
            print(attention[start:arr_chg[1]])
            
    else:
        start = arr_chg[0]+1
        end = arr_chg[1]
        if attention[start] == True and (end-start) >= min_len:
            seq = dataset.getoriginalseq(index)[0][start:end]
        else:
            return "", ""
            print(attention[start:arr_chg[1]])

    return [start, arr_chg[1]], seq

