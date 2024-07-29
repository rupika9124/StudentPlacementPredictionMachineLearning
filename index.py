import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# Step 1: Load the dataset
data = pd.read_csv("Placement_Data_Full_Class.csv")

# Step 2: Handle missing values if any
data.dropna(inplace=True)

# Step 3: One-hot encode categorical variables
categorical_columns = ['gender', 'ssc_b', 'hsc_b', 'hsc_s', 'degree_t', 'workex', 'specialisation']
data_encoded = pd.get_dummies(data, columns=categorical_columns)

# Step 4: Split data into features and target variable
X = data_encoded.drop(['status', 'salary'], axis=1)
y = data_encoded['status']

# Step 5: Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 6: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 7: Initialize and train the Random Forest classifier
rf_classifier = RandomForestClassifier()
svm_classifier = SVC(probability=True)
knn_classifier = KNeighborsClassifier()

rf_classifier.fit(X_train, y_train)
svm_classifier.fit(X_train, y_train)
knn_classifier.fit(X_train, y_train)


# Step 8: Make predictions
rf_predictions = rf_classifier.predict(X_test)
svm_predictions = svm_classifier.predict(X_test)
knn_predictions = knn_classifier.predict(X_test)

def evaluate_model(predictions, y_true):
    accuracy = accuracy_score(y_true, predictions)
    precision = precision_score(y_true, predictions)
    recall = recall_score(y_true, predictions)
    return accuracy, precision, recall

# Step 9: Evaluate the performance of the classifier
rf_metrics = evaluate_model(rf_predictions, y_test)
svm_metrics = evaluate_model(svm_predictions, y_test)
knn_metrics = evaluate_model(knn_predictions, y_test)

print("Random Forest Classifier:")
print("Accuracy:", rf_metrics[0])
print("Precision:", rf_metrics[1])
print("Recall:", rf_metrics[2])

print("\nSupport Vector Machine Classifier:")
print("Accuracy:", svm_metrics[0])
print("Precision:", svm_metrics[1])
print("Recall:", svm_metrics[2])

print("\nK-Nearest Neighbors Classifier:")
print("Accuracy:", knn_metrics[0])
print("Precision:", knn_metrics[1])
print("Recall:", knn_metrics[2])

# Step 10: Calculate probabilities for ROC curve
rf_probs = rf_classifier.predict_proba(X_test)[:, 1]
svm_probs = svm_classifier.predict_proba(X_test)[:, 1]
knn_probs = knn_classifier.predict_proba(X_test)[:, 1]

# Calculate ROC curve
rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_probs)
svm_fpr, svm_tpr, _ = roc_curve(y_test, svm_probs)
knn_fpr, knn_tpr, _ = roc_curve(y_test, knn_probs)

# Step 11: Calculate ROC AUC score
rf_auc = roc_auc_score(y_test, rf_probs)
svm_auc = roc_auc_score(y_test, svm_probs)
knn_auc = roc_auc_score(y_test, knn_probs)

# Step 12: Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(rf_fpr, rf_tpr, label=f'Random Forest (AUC = {rf_auc:.2f})')
plt.plot(svm_fpr, svm_tpr, label=f'Support Vector Machine (AUC = {svm_auc:.2f})')
plt.plot(knn_fpr, knn_tpr, label=f'K-Nearest Neighbors (AUC = {knn_auc:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

# Step 13: Predict placement for a new student
ssc_p = 56  # Example SSC percentage
hsc_p = 52  # Example HSC percentage
degree_p = 52  # Example degree percentage
etest_p = 66  # Example E-Test percentage
workex = "No"  # Example work experience
mba_p = 59.43  # Example MBA percentage

# One-hot encode workex
workex_encoded = 1 if workex == "Yes" else 0

# Prepare input data for prediction
new_student_data_encoded = pd.DataFrame({'ssc_p': [ssc_p],
                                         'hsc_p': [hsc_p],
                                         'degree_p': [degree_p],
                                         'etest_p': [etest_p],
                                         'workex': [workex_encoded],
                                         'mba_p': [mba_p]})

# Ensure new_student_data_encoded has the same columns as X_encoded
missing_cols = set(X.columns) - set(new_student_data_encoded.columns)
for col in missing_cols:
    new_student_data_encoded[col] = 0

# Reorder columns to match X_encoded
new_student_data_encoded = new_student_data_encoded[X.columns]

# Scale the new student data
new_student_data_scaled = scaler.transform(new_student_data_encoded)

# Make prediction
prediction = rf_classifier.predict(new_student_data_scaled)[0]

if prediction == 1:
    print("The model predicts that the student will get placement.")
else:
    print("The model predicts that the student will not get placement.")
