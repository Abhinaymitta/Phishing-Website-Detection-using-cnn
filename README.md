# Phishing Website Detection using Improved Multilayered CNN (IM-CNN)

## 📌 Introduction

This project, titled **"Phishing Website Detection using Improved Multilayered CNN (IM-CNN)"**, aims to identify phishing websites by analyzing structured website features using a convolutional neural network-based deep learning approach. The model classifies websites as either **legitimate** or **phishing**, based on patterns learned from a Kaggle dataset containing various website attributes.

The proposed IM-CNN architecture improves detection accuracy by learning complex patterns using stacked convolutional layers, dropout regularization, and dense classification layers.




---

## 👤 Developer

**Name**  
Mitta Abhinay

**Co-Developer**  
Kakarla Lakshmi Divya Deepthi

---

## 📊 Dataset

- **Source**: [Kaggle](https://www.kaggle.com/)
- **File**: `phishing_dataset.csv`
- The dataset includes various website-based features such as:
  - URL Length
  - Use of HTTPS
  - Presence of special symbols
  - Domain age
  - Web traffic data
- **Target column**: `Result`  
  - `1` for phishing  
  - `-1` for legitimate

---

## ⚙️ Instructions

This project is implemented using **Google Colab** (recommended) or can be run in a local Jupyter environment.

### 📥 Setup Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/your-Abhinaymitta/phishing-website-detection.git
   cd phishing-website-detection

 ________
**🧠 Coding Tasks Breakdown**
1. Loading the Dataset

    Objective: Load the dataset into a pandas dataframe and inspect its contents.

     Used pandas to load the CSV file containing the dataset.

     Checked for missing values and data types.

     Verified feature types (categorical and numerical).

     Ensured a balanced sentiment distribution for training.

2. Preprocessing the Data

    Objective: Clean and prepare the dataset for model training.

      Removed any redundant or irrelevant columns.

      Normalized numerical features using MinMaxScaler.

      Applied one-hot encoding to categorical features.

      Handled missing values using mean/median imputation (if necessary).

      Split the dataset into features (X) and target labels (y).

3. Splitting the Dataset

    Objective: Split the dataset into training and testing sets.

    Used train_test_split from sklearn.model_selection to split the data into an 80-20 training-testing ratio.

    Ensured random shuffling for better model generalization.

4. Building the Improved Multilayered CNN (IM-CNN) Model

    Objective: Design and train the CNN model.

    Designed a custom CNN architecture using Conv1D layers to extract features from the dataset.

    Applied ReLU activations, MaxPooling1D for dimensionality reduction, and Dropout for regularization.

    Flattened the output and passed it through dense layers before the output layer with Sigmoid activation for binary classification.

    Trained the model using binary cross-entropy loss and the Adam optimizer for 30 epochs with early stopping.

5. Model Evaluation

    Objective: Evaluate model performance on the test set.

    Computed the accuracy, precision, recall, and F1-score.

    Generated and analyzed the confusion matrix to understand true/false positives and negatives.

    Visualized performance metrics like training vs validation accuracy and loss curves.

6. Visualizing Results

    Objective: Present results clearly through plots and graphs.

    Plotted training vs validation accuracy/loss curves to evaluate overfitting.

    Visualized the confusion matrix as a heatmap to understand model performance.
__________
💡 Suggestions for Improvement

   Apply feature selection techniques for dimensionality reduction.

   Experiment with hybrid CNN + RNN or transformer-based models.

   Use ensemble techniques like AdaBoost or Random Forest for better generalization.

   Hyperparameter tuning using GridSearchCV or Optuna.

   Save and deploy the trained model using Flask or FastAPI for real-time usage.
______
📂 Project Structure

```bash
phishing-website-detection/
├── dataset/
│   └── phishing_dataset.csv
├── phishing_detection.py        
├── requirements.txt
├── README.md
└── .gitignore
