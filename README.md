# About
> Data and code for the ICDCIT 2027 short paper, **"Representation-Architecture Tradeoffs in Behavioral Malware Detection", (Srivastava et al., 2026)** that is to appear in the associated Springer LNCS proceedings volume.

This repository contains implementations for sequential and graph-based representation-architectures (models) on the [Windows API call sequences dataset](https://www.kaggle.com/datasets/ang3loliveira/malware-analysis-datasets-api-call-sequences) introduced in the work of [Oliveira & Sassi (2019)](https://www.techrxiv.org/doi/pdf/10.36227/techrxiv.10043099.v1) on behavioral malware detection with DGCNNs. 

The following models are implemented:

**Sequential**
* 1D-CNN
* LSTM
* Transformer (mini)
  
**Graph**
* DGCNN
* GAT

... and each of their adversarially regularized [SGAN](https://arxiv.org/pdf/1606.01583) variants (=**10 models** in total).

All models are implemented in PyTorch.

## Structure
* `data`: datasets - original, OOD corpus, test dataset (30% split)
* `meta`: supplementary files and metadata
  * `api.json`: label to API-call vocabulary dictionary
  * `evasions.csv`: raw evaluation data from evasion simulation experiment
  * `ngrams.csv`: top n-grams identified from malware samples in the training dataset
  * `ood_benign.txt`, `ood_malware.txt`: MD5 hashes of benign and malware samples part of the OOD corpus
  * `vt_cache_ood_benign.json`, `vt_cache_ood_malware.json`: provenance metadata of samples sourced via VirusTotal
* `models`: model implementations
  * `saves`: saved models in `.pt` format (per seed)
  * `src`: source code notebooks
* `reports/raw`: raw Cuckoo Sandbox analysis reports of samples from the OOD corpus 
* `scripts`: scripts used for experiments
  * `constants.py`: useful constants used by scripts in the directory
  * `evasion.py`: evasion simulations
  * `predictor.py`: prediction helper functions for all models
  * `ood.py`: runs inference over OOD corpus
* `preprocessor.py`: extracts API features and preprocesses reports into an OOD dataset (CSV)
* `collector.py`: scrapes reports from a [public sandbox endpoint](https://sandbox.pikker.ee/)
* `verify.py`: verifies the origin/metadata of both malware and goodware samples via VirusTotal

## Contributing
This repository is **not** accepting any contributions / PRs.
