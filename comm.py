print("hi")
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

# Step 1: Load the Dataset
data = pd.read_csv("late.csv")

# Step 2: Data Preprocessing
# Handle categorical columns: Encode Protocol
data['Protocol'] = data['Protocol'].astype('category').cat.codes

# Extract source and destination IPs
# You can further group these by subnet or flag local vs external traffic
data['Source Type'] = data['Source'].apply(lambda x: 'local' if '192.168' in x else 'external')
data['Destination Type'] = data['Destination'].apply(lambda x: 'local' if '192.168' in x else 'external')
data['Source Type'] = data['Source Type'].astype('category').cat.codes
data['Destination Type'] = data['Destination Type'].astype('category').cat.codes

# Convert 'Time' to time delta
data['Time Delta'] = data['Time'].astype(float).diff().fillna(0)

# Encode Info into traffic-specific labels
# This step requires custom mapping based on the dataset's patterns or prior knowledge
data['Traffic Type'] = data['Info'].apply(lambda x: 'Application' if 'Application' in x else 'Other')

# Label encode the traffic type (target variable for classification)
le = LabelEncoder()
data['Traffic Type'] = le.fit_transform(data['Traffic Type'])

# Step 3: Feature Selection
features = ['Length', 'Protocol', 'Time Delta', 'Source Type', 'Destination Type']
X = data[features]
y = data['Traffic Type']

# Step 4: Split Dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Normalize Numerical Features
scaler = StandardScaler()
X_train[['Length', 'Time Delta']] = scaler.fit_transform(X_train[['Length', 'Time Delta']])
X_test[['Length', 'Time Delta']] = scaler.transform(X_test[['Length', 'Time Delta']])

# Step 6: Train a Machine Learning Model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Step 7: Evaluate the Model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))

# Step 8: Predict on New Data
# Simulate new data (use actual data in practice)
new_data = pd.DataFrame({
    'Length': [200, 500],
    'Protocol': [2, 1],  # Encoded values for protocols
    'Time Delta': [0.1, 0.05],
    'Source Type': [1, 0],  # 1=external, 0=local
    'Destination Type': [0, 1],  # 0=local, 1=external
})
new_data[['Length', 'Time Delta']] = scaler.transform(new_data[['Length', 'Time Delta']])
predictions = model.predict(new_data)

print("Predicted Traffic Types:", le.inverse_transform(predictions))
import seaborn as sns
import matplotlib.pyplot as plt
sns.histplot(data['Length'], kde=True)  # Distribution of packet lengths
plt.show()
