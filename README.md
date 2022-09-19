# Algorithmic Trading Bot with SVM and Random Forest

This application was written to explore how different trading signals and its differing parameters can affect the training and fitting of sklearn's SVM and Random Forest ML models when applied to an algorithmic trading bot. Particularly, this algorithm looks to optimize the returns of the SVM bot by finding the optimal training set size and windows for the short and long SMA. A Random Forest model is then used with the original parameters to explore and compare how another ML model will compare with SVM.

---

## Technologies 

This application runs on Python 3.7.

---

## Libraries

Import statements already present in program, refer to them below:

```python
import pandas as pd
import numpy as np
from pathlib import Path
import hvplot.pandas
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from pandas.tseries.offsets import DateOffset
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
```

---

## Database

This program couldn't run without the CSV data:

* emerging_markets_ohlcv.csv

---

## Results

These are the results of the model given the original parameters set: 

![SVM_model_original](https://user-images.githubusercontent.com/107497500/190939998-3dc59d0f-2e22-4d7c-8873-556a8a20339f.png)

The performance is fairly lackluster; there is plenty of room for improvement. Initially, the training set was increased to 24 months. Unfortunately this led to drastic overfitting of the model to the dataset: 

![SVM_model_step1](https://user-images.githubusercontent.com/107497500/190940171-04031d45-cfcf-41a2-b2f4-516ebf644d16.png)

The second step was to alter the rolling short and long windows of the SMAs. Initially a short window of 4 and long window of 100, it was optimized to be a short window of 5 and a long window of 50. The following result is also the fully optimized final model of a 3 month training data set, short SMA window of 5 days, and long SMA window of 50 days. Refer below: 

![SVM_model_step2](https://user-images.githubusercontent.com/107497500/190940285-433d5a1a-e2f2-4db1-8dd9-7f3efa0e4039.png)

The final model generated was a Random Forest model, but it performed 2% less accurate than both baseline and tuned models. Refer to the plot generated below:


![RF_model_final](https://user-images.githubusercontent.com/107497500/190940813-21d37a9a-ec95-4df8-9095-725cc77e45d7.png)

---
## Contributors

Thank you for Eric Cardena for teaching Rice's FinTech Boot Camp. He was instrumental in teaching and helping us understand this material. Thank you for Rice for preparing this curriculum and the corresponding data sets that are required to be used. 

**Rishi Prasadha**

**LinkedIn**: https://www.linkedin.com/in/rishi-prasadha-912212133/

**Instagram**: @therishiprasadha

**Twitter**: @RishiPrasadha

---

## Licenses 

MIT
