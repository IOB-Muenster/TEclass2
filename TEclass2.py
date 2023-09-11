import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils.config as cfg
cfg.load_config()
from utils.config import config

import dnaformer.tokenizer as tokenizer
import utils.dataset as dataset


#define the used tokenizer
kmer_tokenizer = tokenizer.kmer_tokenizer

try:
    if config['database']:
        #create new dataset and saves it
        dataset.create_new_dataset(config['te_db_path'], split=True, save=True)
        exit()
    elif config['train']:
        dataset_train, dataset_valid, dataset_test = dataset.return_datasets(kmer_tokenizer, config['dataset_path'])
        #call after dataset has been created -> for custom loss-weigths of specific dataset
        import dnaformer.dnaformer as dnaformer
        # LOAD MODEL
        if config['from_checkpoint']:
            checkpoint = config['model_save_path'] + config['model_name'] + '/' + config['from_checkpoint']
        else:
            checkpoint = None
        dnaformer.model = dnaformer.get_model()

        dnaformer.train(dataset_train, dataset_valid, checkpoint=checkpoint)
        dnaformer.load_model(config['model_save_path'] + config['model_name'] + '/')
        dnaformer.predict(dataset_test)

    elif config['classification']:
        import dnaformer.dnaformer as dnaformer
        dnaformer.load_model(config['model_save_path'] + config['model_name'] + '/' + config['from_checkpoint'])
        dataset_classification = dataset.return_datasets(kmer_tokenizer, None, dataset.create_new_dataset(config['classification_dataset_path'], split=False), split=False)
        dnaformer.classify(dataset_classification)
except Exception as e:
    print("\n\nError:")
    print(e)
