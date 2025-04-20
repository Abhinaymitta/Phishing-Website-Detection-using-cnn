import numpy as np
import pandas as pd
import hashlib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, Flatten, Dropout, BatchNormalization, MaxPooling1D
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

# ✅ Load Data
# Replace 'phishing_data.csv' with the actual file path
data = pd.read_csv('phishing_data.csv')

# ✅ Verify Data
print(data.head())
print(data.columns)

# ✅ Handle Non-Numeric Columns
# Example: Encoding URL using hashing if it's useful for prediction
if 'url' in data.columns:
    data['url_encoded'] = data['url'].apply(lambda x: int(hashlib.md5(x.encode()).hexdigest(), 16) % 10**8)
    data = data.drop(columns=['url'])  # Drop original URL column after encoding

# ✅ Encode Target Column
encoder = LabelEncoder()
data['status'] = encoder.fit_transform(data['status'])  # Convert categorical target to numeric
print(f"Target mapping: {dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))}")

# ✅ Prepare Data
X = data.drop(columns=['status']).values.astype('float32')  # Ensure float32 data type
y = data['status'].values.astype('float32')  # Convert target to float32

# ✅ Normalize Data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ✅ Reshape input for CNN (samples, timesteps, features)
X_reshaped = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)

# ✅ Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y, test_size=0.2, random_state=42)

# ✅ Ensure Data Type Consistency
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# ✅ Ensure CNN-Compatible Input Shape
print(f"Input Shape: {X_train.shape}")  # Should be (samples, timesteps, features)

# ✅ Build CNN Model
model = Sequential([
    Conv1D(128, kernel_size=5, activation='relu', padding='same', input_shape=(X_train.shape[1], 1)),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    Dropout(0.3),

    Conv1D(256, kernel_size=5, activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    Dropout(0.3),

    Conv1D(512, kernel_size=3, activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    Dropout(0.4),

    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.4),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')  # Binary Classification
])

# ✅ Compile Model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# ✅ Train Model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=64)

# ✅ Evaluate Model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"🚀 CNN Model Accuracy: {test_acc:.4f}")

# ✅ Predict on Test Data
y_pred = (model.predict(X_test) > 0.5).astype("int32")

# ✅ Classification Report
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# ✅ Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Legitimate", "Phishing"], yticklabels=["Legitimate", "Phishing"])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# ✅ ROC Curve
y_prob = model.predict(X_test).ravel()
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, color='blue', label=f'AUC = {roc_auc:.4f}')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.show()
