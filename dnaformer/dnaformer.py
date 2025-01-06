from transformers import TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import torch
import utils.visualization as vis
from utils.config import config
from dnaformer.model_definitions import DNAFormer_Trainer, get_model, get_model_config
from utils.dataset import datadict_ as target_names_
model = None

# define the training arguments
training_args = TrainingArguments(
    output_dir=config['model_save_path'] + config["model_name"],
    num_train_epochs=config["num_train_epochs"],
    per_device_train_batch_size=config["train_batch_size"],
    gradient_accumulation_steps=config["gradient_accumulation_steps"],
    per_device_eval_batch_size=config["eval_batch_size"],
    evaluation_strategy=config["evaluation_and_save_strategy"],
    eval_accumulation_steps=config["eval_accumulation_steps"],
    save_strategy=config["evaluation_and_save_strategy"],
    disable_tqdm=config["disable_tqdm"], 
    dataloader_num_workers=8,
    metric_for_best_model = 'eval_f1',
    load_best_model_at_end=True,
    warmup_steps=config["warmup_steps"],
    weight_decay=config["weight_decay"],
    logging_steps=config["logging_steps"],
    fp16=config["fp16"],  
    logging_dir=config['model_save_path'] + config['model_name'] + '/logs',
    report_to="tensorboard",
    optim="adamw_torch"
)

#check for available devices
if not torch.cuda.is_available():
    print("No cuda devices found, using CPU")


def compute_metrics(pred):
    '''
    Computes metrics for a prediction
    '''
    labels = pred.label_ids
    preds = pred.predictions

    preds = preds.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    report = classification_report(labels, preds, target_names=target_names_.keys())

    vis.save_confusion_matrix(labels, preds)
    metrics = {
        "accuracy": float(acc),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "report": report
    }
    return metrics

def train(dataset_train, dataset_valid, checkpoint=None):
    '''
    Trains and evaluates a model based on the given datasets
    '''
    print('Train')
    sample_weight = dataset_train.sample_weight
    # instantiate the trainer class
    trainer = DNAFormer_Trainer(
        sample_weight = sample_weight,
        model = model,
        args = training_args,
        compute_metrics = compute_metrics,
        train_dataset = dataset_train,
        eval_dataset = dataset_valid
    )
    if checkpoint:
        trainer.train(resume_from_checkpoint=checkpoint)
    else: trainer.train()
    trainer.save_model(config['model_save_path'] + config['model_name'])     #save best model
    metrics = trainer.evaluate()
    trainer.save_metrics('eval', metrics)


def classify(dataset_classification):
    '''
        Visualize important parts on each individual sequence
    '''
    # instantiate the trainer class
    trainer = DNAFormer_Trainer(
        model = model,
        args = training_args
    )
    if not config["low_memory"]:
        predictions, _, _ = trainer.predict(dataset_classification)
        output_logits = predictions[0]
        local_attentions = predictions[1]
        global_attentions = predictions[2]
        vis.visualize_and_save(dataset_classification, output_logits, local_attentions, global_attentions, save_imgs=config["save_vis_imgs"])
    else:
        predictions, _, _ = trainer.predict(dataset_classification)
        vis.generate_tsv(predictions, [], dataset_classification)


def predict(dataset_test):
    '''
        Get the metrics on the test-dataset
    '''
    # instantiate the trainer class
    sample_weight = dataset_test.sample_weight
    trainer = DNAFormer_Trainer(
        sample_weight = sample_weight,   
        model = model,
        args = training_args,
        compute_metrics = compute_metrics
    )
    
    _, _, metrics = trainer.predict(dataset_test)
    trainer.save_metrics('test', metrics)


def load_model(path):
    '''
        Loads a model from path
    '''
    print("Load model from", path)
    global model
    model = get_model()
    model = model.from_pretrained(path, config=get_model_config())

