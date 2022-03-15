# TinyML has a Security Problem - An Adversarial Perturbation Perspective

Watch the presentation video here: https://www.youtube.com/watch?v=XSgr5yvSx0M

Course Project Repository for UCLA ECE 209AS (Secure and Trustworthy Edge Computing Systems), Winter 2022. Members: Swapnil Sayan Saha, Khushbu Pahwa, and Basheer Ammar. Supervisor: Professor Nader Sehatbakhsh.

For any information, contact at swapnilsayan@g.ucla.edu. 

## Goal
Recent advancements in machine learning have opened a new opportunity to bring intelligence to low-end Internet-of-Things nodes for making complex and time-critical inferences from unstructured data. Dubbed TinyML, the advancements include model compression, lightweight machine-learning blocks, AutoML frameworks, and software suites designed to perform ultra-low-power, always-on, and onboard sensor data analytics on resource-constrained platforms. However, the first-generation TinyML workflow does not include attack surface analysis and tools to defend the inference pipeline against attacks at various layers in the cyber-physical system. Moreover, the security cost of using “lightweight by design” models on embedded hardware against adversarial attacks has not been explored.
* We perform two well-known adversarial attacks on 10 state-of-the-art TinyML models and 13 state-of-the-art
large neural networks across three applications. We show that TinyML models are less robust to adversarial
perturbations than large models.
* We propose an efficient neural architecture search (NAS) framework to yield models that have both high
utility and adversarial accuracy within the target platform bounds while adding negligible search and training
costs in the TinyML workflow.

While solutions for certifiable adversarial robustness are well studied in the machine learning community, our focus is on providing a starting point to make the recent advancements in TinyML more backward compatible and functionally equivalent (high fidelity) to upstream models without adding significant training and compute overhead.

## Application and Model Information:
We have three different applications, namely image recognition (dataset: CIFAR 10), audio keyword spotting (dataset: Google Speech Commands), and human activity detection (dataset: AURITUS). We trained several large models and several TinyML models for each applications. Please check our report ```Writeup TinyML.pdf``` and presentation ```Presentation_TinyML.pdf``` for more information. Please watch the presentation video (https://www.youtube.com/watch?v=XSgr5yvSx0M) as well.

## Requirements:

Entire codebase runs on Python 3.8.10 and Jupyter Notebook. Please install the Python requirements shown in ```requirements.py```

## Folder Structure:
* ```Model Training``` has all scripts to train or transfer learn candidate models on target datasets. We also provide the trained models.
* ```Attacks``` perform FGSM and PGD attacks on the pre-trained models in previous folder.
* ```Robust NAS``` showcases our NAS with adversarial robustness cost on Bonsai, ProtoNN and TCN models.




