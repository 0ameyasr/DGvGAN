Malware Classification via GAN-based Anomaly Detection

This repository contains the implementation and experiments for malware detection using Generative Adversarial Networks (GANs). The project focuses on behavioral analysis of programs through their API call sequences, rather than traditional signature-based methods. By training a GAN exclusively on legitimate software (goodware), the system identifies malware as anomalous behavior deviating from the learned norm.

Overview

Traditional malware detection techniques struggle to keep up with the increasing sophistication of threats, particularly zero-day attacks and obfuscated malware. This project addresses this challenge by leveraging GANs for anomaly detection, capturing the behavioral patterns of benign programs and flagging deviations indicative of malicious activity.

Key Features

GAN-based anomaly detection trained on goodware sequences.
Behavioral analysis using API call sequences instead of static signatures.
Anomaly scoring system to classify unseen programs as benign or malicious.
Evaluation metrics designed for imbalanced datasets, including Precision, Recall, F1-Score, and AUC-ROC.

Dataset

The project utilizes the “Malware Analysis Datasets: API Call Sequences” dataset from Kaggle, containing:
42,797 malware samples
1,079 benign samplesAPI call sequences representing program behavior during execution (up to 100 calls per sample)

Dataset link

Applications

This project demonstrates approaches relevant to modern cybersecurity practices:
Next-generation antivirus (NGAV) engines
Automated threat analysis in Security Operations Centers (SOC)
Cloud-based executable scanning for real-time threat detection

Repository Structure

data/ – Preprocessed API call sequences and dataset splits
models/ – GAN architecture and training scripts
notebooks/ – Exploratory data analysis and experiments
evaluation/ – Metrics computation and performance analysis
README.md – Project overview and instructions