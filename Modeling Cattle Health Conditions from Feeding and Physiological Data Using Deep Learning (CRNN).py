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
plt.figure(figsize=(6,5))
sns.scatterplot(
    x=df[numeric_cols[0]],
    y=df[numeric_cols[1]],
    hue=df["Disease_Status"],
    palette="Set2"
)
plt.title("Feature Relationship by Disease Class")
plt.show()
#!pip intall
!pip install ydata-profiling
import pandas as pd
from ydata_profiling import ProfileReport
TARGET_COLUMN = "Disease_Status"   #

X = df.drop(columns=[TARGET_COLUMN])
y = df[TARGET_COLUMN]
# Keep ONLY numeric columns
X = X.select_dtypes(include=["int64", "float64"])
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

num_classes = len(np.unique(y_encoded))
num_classes
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
selector = SelectKBest(
    score_func=mutual_info_classif,
    k=300
)

X_reduced = selector.fit_transform(X_scaled, y_encoded)
X_train, X_test, y_train, y_test = train_test_split(
    X_reduced,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test  = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

X_train.shape

input_shape = X_train.shape[1:]

inputs = Input(shape=input_shape)

# 🔵 CNN BLOCK
x = Conv1D(filters=64, kernel_size=3, activation="relu", padding="same")(inputs)
x = BatchNormalization()(x)
x = MaxPooling1D(pool_size=2)(x)

x = Conv1D(filters=128, kernel_size=3, activation="relu", padding="same")(x)
x = BatchNormalization()(x)
x = MaxPooling1D(pool_size=2)(x)

# 🔵 RNN BLOCK
x = LSTM(128, return_sequences=False)(x)
x = Dropout(0.4)(x)

# 🔵 DENSE HEAD
x = Dense(128, activation="relu")(x)
x = Dropout(0.3)(x)

outputs = Dense(num_classes, activation="softmax")(x)

model = Model(inputs, outputs)
model.summary()