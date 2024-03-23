import pandas as pd
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
import numpy as np
import matplotlib

#I aimed to find relationship between flight hour and if the passenger wanted a meal or not.
df = pd.read_csv('https://raw.githubusercontent.com/egeenc0/Simple-ANN/main/customer_booking.csv')
"""
print(df.head())
print(df.columns)
print(df.dtypes)
"""

x = df['flight_hour']
y = df['wants_in_flight_meals']
x_data = np.array(x)
y_data = np.array(y)

print(x_data)
print(y_data)
#Data is prepared

model = Sequential()

model.add(Dense(64, input_shape=(1,), activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(x_data, y_data, epochs=3)

def predict_val(x):
    return 1 if 0.5 > (model.predict(np.array(x).reshape(1,1))) else 0

sample_data = [t for t in range(10)]
