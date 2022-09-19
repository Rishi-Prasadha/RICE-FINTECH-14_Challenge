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