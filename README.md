# Machine Learning Project: Tesla Stock Price Prediction

## Overview
This project focuses on predicting Tesla's stock price movement (up or down) using historical stock data. The dataset spans from 2015 to 2025 and includes features such as opening price, highest price, lowest price, closing price, and trading volume. The goal is to classify whether the stock price will increase or decrease on the next trading day.

## Objectives
- Preprocess Tesla stock price data for machine learning tasks.
- Build and evaluate classification models to predict stock price movement.
- Analyze the performance of different models and visualize the results.

---

## Website Visualization

A simple and attractive website is provided to showcase the comparison results of the three machine learning models (Decision Tree, Naive Bayes, KNN) used in this project. The website displays the accuracy and AUC of each model, along with a summary and a comparison chart.

**How to use:**
1. Buka folder `tesla-ml-model-comparison` (atau sesuai nama folder website).
2. Jalankan file `index.html` di browser favorit Anda.
3. Anda akan melihat tampilan perbandingan model beserta grafik visualisasi.

---

## Workflow

### 1. **Data Preprocessing**
The raw dataset (`TSLA_Preprocessed.csv`) contains daily stock price data. The preprocessing steps include:
- **Cleaning Numeric Values**: Removing commas and converting columns to numeric types.
- **Date Conversion**: Converting the `date` column to a datetime format and sorting the data chronologically.
- **Feature Engineering**: Creating a new target column `Price_Up`:
  - `Price_Up = 1` if the closing price increased compared to the previous day.
  - `Price_Up = 0` otherwise.
- **Handling Missing Values**: Dropping rows with missing values caused by the `diff()` operation.

The preprocessed dataset is saved as `TSLA_Preprocessed.csv`.

---

### 2. **Algorithms and Models**
Three machine learning models were implemented and evaluated:

#### **Decision Tree Classifier**
- **Library**: Scikit-learn
- **Parameters**: `max_depth=5`, `random_state=42`
- **Workflow**:
  1. Split the dataset into training and testing sets (80% train, 20% test).
  2. Train the Decision Tree model on the training set.
  3. Predict stock price movement on the test set.
  4. Evaluate the model using metrics such as accuracy, precision, recall, F1-score, and AUC.

#### **Naive Bayes Classifier**
- **Library**: Scikit-learn
- **Workflow**:
  1. Similar to the Decision Tree workflow.
  2. Predict probabilities for ROC curve analysis.

#### **K-Nearest Neighbors (KNN) Classifier**
- **Library**: Scikit-learn
- **Parameters**: `n_neighbors=5`, `metric='minkowski'`, `p=2` (Euclidean distance)
- **Workflow**:
  1. Normalize the dataset to ensure all features are on the same scale.
  2. Split the dataset into training and testing sets (80% train, 20% test).
  3. Train the KNN model using the training set.
  4. Predict stock price movement on the test set.
  5. Evaluate the model using metrics such as accuracy, precision, recall, F1-score, and AUC.

---

### 3. **Evaluation Metrics**
The models were evaluated using the following metrics:
- **Confusion Matrix**: Visualizes true positives, true negatives, false positives, and false negatives.
- **Classification Report**: Includes precision, recall, F1-score, and support for each class.
- **ROC Curve and AUC**: Measures the model's ability to distinguish between classes.

---

## Results

### **Decision Tree Classifier**
- **Accuracy**: 61%
- **Precision**: Higher for predicting "Naik" (up) than "Turun" (down).
- **AUC Score**: 0.6451
- **Insights**:
  - The model performed well in predicting upward movements but struggled with downward movements.
  - The decision tree structure was visualized to understand feature importance.

### **Naive Bayes Classifier**
- **Accuracy**: 52.8%
- **AUC Score**: 0.6049
- **Insights**:
  - The model showed moderate performance but was less effective than the Decision Tree.
  - It struggled with imbalanced class predictions.

### **K-Nearest Neighbors (KNN) Classifier**
- **Accuracy**: 58.3%
- **AUC Score**: 0.6217
- **Insights**:
  - The KNN model performed better than Naive Bayes but slightly worse than the Decision Tree.
  - The choice of `k` (number of neighbors) significantly impacted the model's performance.
  - Normalization improved the model's accuracy by ensuring all features contributed equally to distance calculations.

---

## Analysis and Insights
- **Feature Importance**: The closing price and trading volume were significant predictors of stock movement.
- **Model Comparison**: The Decision Tree outperformed both Naive Bayes and KNN in terms of accuracy and AUC.
- **Challenges**:
  - Imbalanced classes (more "Turun" than "Naik") affected model performance.
  - Stock price prediction is inherently noisy and influenced by external factors not captured in the dataset.

---

## Visualization
- **Confusion Matrix**: Displayed for all models to analyze prediction errors.
- **ROC Curve**: Plotted to compare the true positive rate and false positive rate across models.
- **Decision Tree Structure**: Visualized to understand the decision-making process.
- **Website**: Interactive web visualization for model comparison.

---

## Conclusion
This project demonstrates the application of machine learning to predict stock price movements. While the Decision Tree model showed the best results, the KNN model also performed reasonably well after normalization. Further improvements could be made by:
- Incorporating additional features (e.g., macroeconomic indicators, news sentiment).
- Using ensemble methods like Random Forest or Gradient Boosting.
- Addressing class imbalance through oversampling or cost-sensitive learning.

---

## How to Run the Project
1. Clone the repository.
2. Install the required Python libraries:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the preprocessing script or notebook to generate the preprocessed dataset.
4. Train and evaluate models using the provided Jupyter Notebooks:
   - `DecisionTree_Histogram.ipynb`
   - `NaiveBayes_Histogram.ipynb`
   - `KNN_Histogram.ipynb`
5. **(Optional) Buka website perbandingan model:**
   - Buka folder website dan jalankan `index.html` di browser.

---

## Files in the Repository
- **`requirements.txt`**: Required Python libraries.
- **`TSLA_2015_2025_Histogram.csv`**: Raw dataset.
- **`TSLA_Preprocessed.csv`**: Preprocessed dataset.
- **`DecisionTree_Histogram.ipynb`**: Notebook for Decision Tree model.
- **`NaiveBayes_Histogram.ipynb`**: Notebook for Naive Bayes model.
- **`KNN_Histogram.ipynb`**: Notebook for KNN model.
- **`Tabel 3 model.pdf`**: Analysis results of the three models.
- **`tesla-ml-model-comparison/`**: Website folder for model comparison visualization.
- **`README.md`**: Project documentation.

---

## Future Work
- Experiment with advanced models like LSTM for time-series forecasting.
- Incorporate external data sources for better feature representation.
- Optimize hyperparameters using grid search or random search.

---

## Authors
**Developed by :**
**Muhamad Nur Arif**
**(41523010147)**

### 🔗 Link
[![portfolio](https://img.shields.io/badge/my_portfolio-000?style=for-the-badge&logo=ko-fi&logoColor=white)](https://arifsuz.vercel.app/)
[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/arifsuz)
[![linkedin](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/marif8/)
[![instagram](https://img.shields.io/badge/Instagram-E4405F?style=for-the-badge&logo=instagram&logoColor=white)](https://www.instagram.com/ariftsx/)