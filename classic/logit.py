import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.cross_validation import  cross_val_score

dta = sm.datasets.fair.load().data
dta['affair'] = (dta.affairs>0).astype(int)

