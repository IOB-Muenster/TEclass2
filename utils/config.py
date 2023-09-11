import sys
import os
import yaml
import argparse

global config

parser = argparse.ArgumentParser(prog='TEclass2', 
                                 description='Software for training and classifing TE sequences using the machine learning model Transformer', 
                                 epilog='' )
group = parser.add_mutually_exclusive_group(required=True)

group.add_argument('--database', dest='database', action='store_true',
                    default=False, 
                    help='builds a database for training a model using the file and categories from the configuration file')

group.add_argument('--train', dest='train', action='store_true',
                    default=False, 
                    help='trains a model reading from the configuration file')

group.add_argument('--classify', dest='classification', action='store_true',
                    default=False, 
                    help='classifies a fasta file taking the model and setup from the configuration file')
parser.add_argument('-f', dest='file_path', action='store',
                    help='define the fasta file to classify.')

parser.add_argument('-o', dest='target_path', action='store',
                    help='define the target folder to store results.')

parser.add_argument('-c', dest='config_path', action='store',
                    default='config.yml',
                    help='define the used config file.')


cmd_args = parser.parse_args()
if cmd_args.classification and (cmd_args.file_path is None or cmd_args.target_path is None):
    parser.error("--classify requires the arguments -f and -o ")
    sys.exit()
 
def check_paths(config):
    if cmd_args.classification:
        file_path = os.path.exists(cmd_args.file_path)
        model_path = os.path.exists(config['model_save_path'] + config['model_name'])
        if not model_path:
            raise SystemExit("Error: Model path does not exist.\nCheck the parameter model_name in the configuration file.")
        if not file_path:
            raise SystemExit("Error: File for classification does not exist.")
    elif cmd_args.train:
        dataset_path = os.path.exists(config['dataset_path'] + "-train.pkl")
        if not dataset_path:
            raise SystemExit("Error: Dataset  path does not exist.\nCheck the parameter dataset_path in the configuration file.")
    elif cmd_args.database:
        te_db_path = os.path.exists(config['te_db_path'])
        if not te_db_path:
            raise SystemExit("Error: Path to file with TEs does not exist.\nCheck the parameter te_db_path in the configuration file.")

def load_config():
    global config
    with open(cmd_args.config_path, 'r') as file:
        config = yaml.safe_load(file)

    config["classification_dataset_path"] = cmd_args.file_path
    config["vis_save_path"] = cmd_args.target_path
    config["train"] = cmd_args.train
    config["classification"] = cmd_args.classification
    config["prediction"] = False
    config["database"] = False
    if config["train"]: 
        config["prediction"] = True
    elif cmd_args.database:
        config["database"] = True
        config["prediction"] = True
        config["train"] = False
    check_paths(config)
