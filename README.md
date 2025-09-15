# DGvGAN - Deep Graph Convolutional Neural Networks v/s Generative Adversarial Networks for Dynamic Malware Analysis

This repository contains an implementation of deep graph convolutional neural network (DGCNN) models for behavioral malware detection, based on [this](https://www.researchgate.net/publication/336989176_Behavioral_Malware_Detection_Using_Deep_Graph_Convolutional_Neural_Networks) paper by **Oliveira & Sassi (2019)**. It also includes comparitive analysis with a **Generative Adversarial Network (GAN)** modeled as an anomaly detector in benign goodware samples to study both approaches for performing dynamic malware analysis and detection. The project focuses on behavioral analysis of programs through their API call sequences, rather than traditional image-translation or conventional signature-based methods. By training a GAN exclusively on legitimate software (goodware), the system aims to identify malware as anomalous behavior deviating from the learned norm. 

## Overview
Traditional malware detection techniques struggle to keep up with the increasing sophistication of threats, particularly zero-day attacks and obfuscated malware. Additionally, paradigms that convert malware binaries to grayscale images tend to be expensive in real-world deployments, and risk misclassifications with goodware samples due to loss of granular information. This project addresses this challenge by first implementing a standard baseline of a DGCNN model and then leveraging GAN architecture for anomaly detection, capturing the behavioral patterns of benign programs and flagging deviations indicative of malicious activity. We hope to study how both processes are compared.

## Key Features
* Implementation of the reference paper's DGCNN models
* Implementation of a GAN for anomaly detection in goodware samples
* Comparitive analysis of all implemented model architectures

## Dataset 
The project utilizes the **“Malware Analysis Datasets: API Call Sequences”** dataset from **Kaggle**, containing:
* **42,797** _malware_ samples, and **1,079** _benign_ samples
* API call sequences representing program behavior during execution (up to 100 calls of 306 total per sample)
  
View the dataset [here](https://www.kaggle.com/datasets/ang3loliveira/malware-analysis-datasets-api-call-sequences/data).

## Applications
This project demonstrates approaches relevant to modern cybersecurity practices:
* Next-generation antivirus (NGAV) engines
* Automated threat analysis in Security Operations Centers (SOC)
* Cloud-based executable scanning for real-time threat detection

## Repository Structure
The repository is structured as follows:
* `data/` – Unprocessed and pre-processed data taken from the dataset
* `scripts/` – Helper scripts to interact with the data and the models
* `models/` - Implementation of each model
