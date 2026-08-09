# About
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
* `meta`: supplementary files and metadata
* `models`: model implementations
  * `saves`: saved models in `.pt` or `.pth` format
  * `src`: source code
* `pipeline`: inference pipeline on a hold-out set of more recent malware/goodware
  * `constants.py`: pipeline constants
  * `evasion.py`: evasion simulations
  * `predictor.py`: prediction helper functions for all models
  * `run.py`: runs the pipeline to infer on the hold-out set
  * `sanity.py`: few sanity checks (experimental)
* `reports`: raw + processed behavioral analysis reports of Windows PE malware/goodware gathered from a [public Cuckoo Sandbox endpoint](https://sandbox.pikker.ee)
* `dataset.csv` and `test.csv`: original dataset and 30% test split used for in-distribution training/testing respectively
* `etc.py`: a file for temporary tests/code/small experiments
* `extractor.py`: extracts API features from raw behavioral analysis reports
* `maintainer.py`: scrapes reports from the public sandbox endpoint
* `provenance.py`: verifies the origin/metadata of both malware and goodware samples via VirusTotal

## Contributing
This repository is **not** accepting any contributions / PRs.
