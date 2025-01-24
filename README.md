# **Amazon Review Sentiment Analysis**

This project demonstrates how to analyze the sentiment of Amazon reviews using machine learning techniques. The process includes text preprocessing, model training, evaluation, and saving the models for deployment.
streamlit: https://sentiment-analysis-amazonreviews.streamlit.app/
## **Overview**

This notebook covers the complete workflow for Amazon review sentiment classification, including:
- Data preprocessing: Cleaning text, tokenization, stopword removal, lemmatization, and stemming.
- Training multiple models: Logistic Regression, Decision Trees, and Support Vector Machine (SVM).
- Addressing data imbalance using SMOTE (Synthetic Minority Over-sampling Technique).
- Evaluating model performance with accuracy and confusion matrices.
- Saving trained models using `pickle` for future use.

The goal of this project is to classify Amazon reviews as either positive or negative based on the review content.

## **Dataset**

This analysis uses the **Amazon Alexa review dataset**, which contains user reviews of Amazon Alexa devices. The target variable for sentiment classification is the **feedback** column.

## **Steps Covered**

### 1. **Data Preprocessing**
- Handling missing values and duplicate rows.
- Text cleaning: converting to lowercase, removing special characters.
- Tokenization and stopword removal.
- Lemmatization and stemming for text normalization.

### 2. **Model Training**
- **Logistic Regression**  
- **Decision Tree Classifier**  
- **Support Vector Machine (SVM)**  
- Evaluation of model performance using accuracy scores and confusion matrices.

### 3. **Addressing Imbalanced Data**
- Use of **SMOTE** to balance the dataset by creating synthetic samples of the minority class.

### 4. **Model Evaluation**
- Performance metrics including accuracy, confusion matrix, and classification report.
- Grid Search for hyperparameter tuning of the SVM model.

### 5. **Saving Models**
- Saving the trained models (Logistic Regression, Decision Tree, SVM) and transformers (TF-IDF vectorizer) using `pickle`.
