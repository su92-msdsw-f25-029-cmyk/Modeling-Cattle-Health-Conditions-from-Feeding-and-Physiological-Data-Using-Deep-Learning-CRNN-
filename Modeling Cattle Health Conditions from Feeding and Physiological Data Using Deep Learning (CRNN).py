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
plt.figure(figsize=(12,5))
df[numeric_cols[:5]].boxplot()
plt.title("Boxplot of Selected Numerical Features")
plt.xticks(rotation=45)
plt.show()
plt.figure(figsize=(6,5))
plt.scatter(
    df[numeric_cols[0]],
    df[numeric_cols[1]],
    alpha=0.6
)
plt.xlabel(numeric_cols[0])
plt.ylabel(numeric_cols[1])
plt.title(f"{numeric_cols[0]} vs {numeric_cols[1]}")
plt.show()
plt.figure(figsize=(10,8))
corr = df[numeric_cols].corr()

sns.heatmap(
    corr,
    cmap="coolwarm",
    linewidths=0.5
)
plt.title("Correlation Heatmap of Numerical Features")
plt.show()
sns.pairplot(
    df,
    vars=numeric_cols[:4],
    hue="Disease_Status"
)
plt.show()