# TEclass2

Repository for TEclass2 a TEs models classification into superfamilies software based in the machine model Transformer originally described by [Vaswani et al](https://arxiv.org/abs/1706.03762).
An online version of TEclass2 is available [here]( https://bioinformatics.uni-muenster.de/tools/TEclass2/index.pl?lang=en). It allows the classification of TEs models into the following superfamilies: Copia, Crypton, ERV, Gypsy, hAT, Helitron, Jockey, L1_L2, Maverick, Merlin, SINE, PIF, Pao, RTE, TcMar and Transib with a weighted average F1-score of 0.78.

TEclass2 can be used for training and building Transformer models based on k-mer frequencies and distribution in DNA sequences of TE models. After training a model it can be used for classifying other unknown TEs  models.
A detailed workflow and the results obtained with this software can be found in the following [publication.](https://www.biorxiv.org/content/10.1101/2023.10.13.562246v1)

## Requirements
For running the software, particularly for building TE training models a GPU is required.
The software requires a [Conda environment]( https://conda.io) and the following packages must be installed (a Conda environment.yml file is provided to facilitate the installation)
The software was tested with Python 3.10.9 and the following libraries supplied by Conda 23.1.0 (recommended) but can also be provided with Python´s pip.
## Installation

### Local
```
git clone https://github.com/IOB-Muenster/TEclass2.git
cd TEclass2
conda env create -f environment.yml
conda activate TEclass2
```

### Docker
The files used are under `./volume/`.
```
git clone https://github.com/IOB-Muenster/TEclass2.git
cd TEclass2
docker build -t teclass2 -f Dockerfile
```



## Running 

### Notes for Docker

Change and/or copy the files into a separate folder and mount these, such as:

```
docker run --rm -v ./volume/:/volume teclass2:latest -c /volume/config.yml --database
docker run --rm -v ./volume/:/volume teclass2:latest -c /volume/config.yml --train
```


### Preparing the dataset for training
For training, the dataset must be a FASTA file with labeled TE models, e.g.
> \>ALUY SINE<br>
> GGCCGGGCGCGGTGGCTCACGCCTGTAATCC...<br><br>
> \>L1M5 LINE<br>
> ATGGTAGATTTAAACCCAANCATATCAATAT...

In this example, SINE and LINE are the labels that are going to be read by the software and it must be specified in the configuration file. It is recommended to provide at least a few thousand TEs from each category of TE for training the model.

### Configuration of the model in config.yml
The configuration file provided config.yml gives an example of all the parameters that are required for running and how they should be provided. It is required to provide the basic configuration such as paths and names, name of the labels provided in the FASTA file, and the configuration for training the model.
Example of configuration of config.yml file and the most important and basic parameters:
```
model_name: 'model_TEs'
model_save_path: './models'
te_keywords: ['SINE', 'LINE', 'LTR', 'DNA']
te_db_path: 'data/TEs_4_categories.fasta
```
**Important:** After building a model using a custom dataset, the same configuration file must be provided to run the classification stage, so it's important to keep a copy of it.

### Building a database
There plenty of other parameters that are important for building and training the model. After all this parameters are set, a database ready for training can be built with the following command:
```
python TEclass2.py –-database –c config.yml
```
### Training a model
If config.yml is correctly configured and the GPU memory is enough for the job, the training can start with:
```
python TEclass2.py –-train –c config.yml
```
While running the training it outputs for every epoch the F1 score for each class and the weighted F1 score for the whole model as the precision and recall scores. The classification step depends on the GPU, the size of the database and the number of labels used, and can take from hours to days.
### Classifying TEs
After training a model it can be used for classification with the following command. The output is a file with the TE ids and their classification, and the Softmax score for each label of TE in the model. The predicted TE label is the one with the highest score.
```
python TEclass2.py –-classify –c config.yml –o outfile
```
The Softmax score is a good indicator of the accuracy of the classification, as scores higher than 0.7 are highly accurate.
Example of the classification output for a model that uses four TE classes.
```
                   SINE    LINE    LTR     DNA
DF0282269.1   LTR  0.000   0.071   0.928   0.001
ZMCOPIA1_I    LTR  0.000   0.016   0.983   0.001
Mariner1N_LA  DNA  0.000   0.000   0.000   1.000
MuDR-5_SBi    DNA  0.001   0.040   0.014   0.945
DF0280483.1   DNA  0.000   0.000   0.000   1.000
```


## Citation
@article {TEClass2,
	author = {Bickmann, Lucas and Rodriguez, Matias and Jiang, Xiaoyi and Makalowski, Wojciech},
	title = {TEclass2: Classification of transposable elements using Transformers},
	elocation-id = {2023.10.13.562246},
	year = {2023},
	doi = {10.1101/2023.10.13.562246},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2023/10/16/2023.10.13.562246},
	eprint = {https://www.biorxiv.org/content/early/2023/10/16/2023.10.13.562246.full.pdf},
	journal = {bioRxiv}
}

