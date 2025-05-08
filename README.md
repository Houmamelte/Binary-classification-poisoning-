# Binary Classification Poisoning

This project has the following steps and each part has seperate documentation as provided below

- **Poisoning** -  [See poisoning README](poisoning/poisoning_README.md)
- **Pre-processing** - [See pre-processing README](Training/Preprocessing_README.md)
- **Model Training** - [See model training README](Training/Training_Readme.md)
- **Evaluation** - [See evaluation README](Evaluation/Evaluation_README.md)

# Literature Review
Data Poisoning involves inserting carefully crafted samples to blend with benign data and evade anomaly detectionmechanisms. Attackers may modify existing samples or insert synthetic data to manipulate the training distribustion.The main approaches of data poisoining include heuristic-based attacks,label flipping, feature collision attacks, bilevel optimization,influence-based methods, and generative model-based attacks("Data Poisoning in Deep Learning: A Survey," 2025).Based on this one of the approaches will be used to poison the data label flipping, which is one is referred to as availability poisoning.The effect of label flipping has been investigated revealed that deep models can perfectly fit training data even when labels are completely randomized,achieving zero training error while failing to generalize on test data. This overfitting behavior demonstrates that deep learning models do not inherently distinguish between correct and incorrect labels, making them highly susceptible to label flipping attacks. 
In a reaearch done on determining the effect of targeted attacks on audio authentication systems 2 data attacks were investigated:backdoor-triggered attacks and targeted data poisoning attacks.Backdoor attack involves implanting imperceptible triggers into training data, allowing adversaries to manipulate outputs whenever these triggers appear("Backdoor Approach with Inverted Labels Using Dirty Label-Flipping Attacks," n.d.). 
The purpose of the backdoor poisoning is to make the model learn to associate the posioned samples with the same pattern as the benign cases, even if it appears in future malignant inputs thus misleading the model performance.("Data Poisoning Attacks in the Training Phase of Machine Learning Models: A Review," n.d.). The main puurpose od data poisoning is to make it the model ineffective in making sound and reliable decisions as outputs generated are inaccurate.


# References:
-A Backdoor Approach with Inverted Labels Using Dirty Label-Flipping Attacks. (2023). arXiv.org e-Print archive. https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10483076
-Boosting Backdoor Attack with A Learnable Poisoning Sample Selection Strategy. (2023, July 14). arXiv.org e-Print archive. https://arxiv.org/pdf/2307.07328
-Data Poisoning Attacks in the Training Phase of Machine Learning Models: A Review. (2024, December 9). CEUR-WS.org - CEUR Workshop Proceedings (free, open-access publishing, computer science/information systems). https://ceur-ws.org/Vol-3910/aics2024_p10.pdf
-Data Poisoning in Deep Learning: A Survey. (2025, March 27). arXiv.org e-Print archive. https://arxiv.org/pdf/2503.22759
-Deep Learning--based Text Classification: A Comprehensive Review. (2021, April 17). https://dl.acm.org/doi/10.1145/3439726
-Emadmakhlouf. (2024, June 13). Neural network decision boundary visualization. Kaggle: Your Machine Learning and Data Science Community. https://www.kaggle.com/code/emadmakhlouf/neural-network-decision-boundary-visualization
-A novel focal-loss and class-weight-aware convolutional neural network for the classification of in-text citations. (2021, March 24). https://journals.sagepub.com/doi/10.1177/0165551521991022
-OWASP Machine Learning Security Top 10 - Draft release v0.3. (2023). https://mltop10.info/OWASP-Machine-Learning-Security-Top-10.pdf
-Poisoning Attacks Against Machine Learning: Can Machine Learning Be Trustworthy? (2024, October 24). National Institute of Standards and Technology. https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=934932
-Towards Class-Oriented Poisoning Attacks Against Neural Networks. (2022). CVF Open Access. https://openaccess.thecvf.com/content/WACV2022/papers/Zhao_Towards_Class-Oriented_Poisoning_Attacks_Against_Neural_Networks_WACV_2022_paper.pdf
-Transferable Availability Poisoning Attacks. (2024, June). arXiv.org e-Print archive. https://arxiv.org/pdf/2310.05141

