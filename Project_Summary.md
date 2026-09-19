# Project Summary: Customer Segmentation using Machine Learning

**Type:** End-to-end data analytics / ML project
**Tools:** Python, pandas, scikit-learn, Streamlit

## Objective
Segment retail customers into distinct behavioural groups using unsupervised machine learning, so that marketing and engagement strategies can be tailored per segment instead of applied uniformly.

## What Was Done
1. **Data cleaning** — loaded a 2,240-row customer marketing dataset, checked for and removed missing values, and corrected data types (e.g., converting enrollment date to datetime).
2. **Feature engineering** — created derived features including customer age, total household children, total spending across product categories, tenure since enrollment, a binary campaign-acceptance flag, and age-group bins.
3. **Exploratory data analysis** — analyzed distributions of age, income, and spending; compared income and spending across education and marital status; built a correlation heatmap and pivot tables to surface relationships between demographics, spending, and purchase behaviour.
4. **Clustering model** — scaled features with `StandardScaler`, used the elbow method (WCSS across K = 2–9) to select the optimal number of clusters, then applied **K-Means (K = 6)** to group customers by age, income, spending, purchase channel activity, and recency.
5. **Dimensionality reduction & visualization** — applied PCA to project the clustering features into 2 dimensions for a clear visual separation of the 6 segments.
6. **Cluster profiling** — computed per-cluster averages to characterize each segment (e.g., a high-income/high-spend, store-preferring segment vs. a low-income/low-engagement segment).
7. **Model deployment prep** — saved the trained K-Means model and scaler with `joblib` for reuse.
8. **Interactive dashboard** — built a Streamlit app to visualize the customer segments interactively, making the analysis explorable without needing to open the notebook.

## Result
Six clearly differentiated customer segments were identified, ranging from high-income/high-spend/store-preferring customers to low-income/low-engagement customers — giving a practical basis for targeted marketing decisions.

## Skills Demonstrated
- Data cleaning and feature engineering (pandas, NumPy)
- Exploratory data analysis and visualization (matplotlib, seaborn)
- Unsupervised machine learning: K-Means clustering, elbow method for model selection
- Dimensionality reduction with PCA
- Model persistence with joblib
- Building an interactive dashboard with Streamlit to communicate analytical results
