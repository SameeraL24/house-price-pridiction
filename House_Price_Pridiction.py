import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_squared_error

# Load the dataset
df = pd.read_csv('Housing.csv')
print(df.head(),"\n")
print(df.info(),"\n")
print(df.describe(),"\n")
print(df.isnull().sum(),"\n")

# Data Preprocessing        
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
cols_to_encode = ['mainroad', 'guestroom', 'basement','hotwaterheating', 'airconditioning', 'prefarea','furnishingstatus']
for col in cols_to_encode:
    df[col] = le.fit_transform(df[col])

print("Data converted using Label Encoding!\n")
print(df.head())
# Features and Target
X = df.drop('price', axis=1)
y = df['price'] 
# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Train the model           
model = LinearRegression()
model.fit(X_train, y_train) 
print("Model trained successfully!\n")


# Predict
y_pred = model.predict(X_test)

# Evaluate
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"R2 Score: {r2}")
print(f"RMSE: {rmse}")

# Test with new data
new_data = [[2500, 3, 2, 1, 1, 0, 1, 1, 0, 1, 1, 0]]  # Example feature values
predicted_price = model.predict(new_data)
print(f"Predicted Price: {predicted_price[0]}")
print("Prediction for new data completed!\n")


# Save the model using pickle   
import pickle
with open('house_price_model.pkl', 'wb') as f:
    pickle.dump(model, f)   
print("Model saved as house_price_model.pkl")

# Visualizations
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'red', lw=2)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted Prices')
plt.show()
print("Visualizations completed!")