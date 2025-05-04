# Data Poisoning

This Section describes the 2 types of data poisoning carried out in the dataset.

## Overview

The two types of data poisoning carried out are backdoor and availability poisoning.

## Dependencies
- pandas
- sklearn
- numpy

## Data Sources

- Clean dataset: `../data/balanced_dataset.csv`
- Poisoned dataset: `../poisoning/poisoned10p.csv` (Or any one you want)
- Poisoned dataset: `../data`
## Data Poisoning Steps
### Availability Poisoning:
- Availability poisoning involves flipping target labels. The data is poisoned through various poisoning ratios (0.1-0.9).​
- The availability poisoning is  in 2 ways:​

  - Changing samples of class 0 to class 1.​
  - Randomly changing class labels for both target values.​
    
### Backdoor Attack:
- This involves introducing a trigger in the comments.
- In our backdoor attack, we randomly inserted the word 'charger'. The percentage poison percentage was 15% of the total number of comments.

## Conclusion
- The poisoned data is saved in the data folder. The various data poiaoning percentages of the availability poisoning will be used to investigate its impact on the binary model performance.
