import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras import layers, models

# Sample guest data
data = {
    'guest_id': [1, 2, 3, 4, 5],
    'previous_stays': [10, 5, 3, 8, 12],
    'spa_visits': [2, 3, 0, 1, 4],
    'dining_visits': [5, 2, 1, 3, 5],
    'preferred_cuisines': ['Italian', 'Chinese', 'Mexican', 'Italian', 'Japanese'],
    'room_preference': ['Suite', 'Single', 'Double', 'Suite', 'Suite']
}

df = pd.DataFrame(data)

# Label encode categorical features
le = LabelEncoder()
df['preferred_cuisines_encoded'] = le.fit_transform(df['preferred_cuisines'])
df['room_preference_encoded'] = le.fit_transform(df['room_preference'])

# Define features and target
X = df[['previous_stays', 'spa_visits', 'dining_visits', 'preferred_cuisines_encoded']].values
y = df['room_preference_encoded'].values

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the model
model = models.Sequential([
    layers.Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dense(16, activation='relu'),
    layers.Dense(1, activation='linear')  # Output layer with a linear activation for regression
])

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=4, validation_split=0.2)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.2f}")
