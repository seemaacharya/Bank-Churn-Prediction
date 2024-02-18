import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

# Load the CSV file
df = pd.read_csv("Churn_Modelling.csv")

# Convert categorical columns (Geography, Gender) into numerical
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
df['Geography'] = label_encoder.fit_transform(df['Geography'])
df['Gender'] = label_encoder.fit_transform(df['Gender'])

df1 = df.drop(columns=['RowNumber', 'CustomerId', 'Surname'])

# Separate independent and dependent variables
X = df1.drop(columns=["Exited"])
y = df1["Exited"]

# Split the dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Instantiate the model
model = Sequential()
model.add(Dense(11, activation='relu', input_dim=10))
model.add(Dense(11, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Fit the model
model.fit(X_train, y_train, batch_size=50, epochs=100, verbose=1, validation_data=(X_test, y_test))

# Save the model to a pickle file
with open("model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)
