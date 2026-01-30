import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix
)
from sklearn.feature_selection import SelectKBest, mutual_info_classif

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D,
    LSTM, Dense, Dropout, BatchNormalization
)
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
df = pd.read_csv("/content/global_cattle_disease_detection_dataset.csv")
df.head()
df.head()
df.info()
df.describe()
numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
numeric_cols
plt.figure(figsize=(7,5))
plt.hist(df[numeric_cols[0]], bins=30, edgecolor="black")
plt.title(f"Distribution of {numeric_cols[0]}")
plt.xlabel(numeric_cols[0])
plt.ylabel("Frequency")
plt.show()