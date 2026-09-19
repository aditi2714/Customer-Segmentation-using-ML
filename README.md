# Customer Segmentation using Machine Learning

An end-to-end customer segmentation project that cleans and explores marketing-campaign data, engineers behavioural features, and applies K-Means clustering (with PCA visualization) to group customers into distinct segments. Results are explored interactively through a Streamlit dashboard.

## Project Overview
Businesses rarely treat all customers the same way — some are high-value frequent buyers, others are price-sensitive occasional shoppers. This project segments customers from a retail marketing dataset into behaviourally distinct groups so that marketing, offers, and outreach can be tailored to each segment rather than applied uniformly.

## Dataset
- **File:** `customer_segmentation.csv`
- **Size:** 2,240 customers, 29 original columns
- **Contents:** demographics (birth year, education, marital status, income, household composition), enrollment date, recency, spend across 6 product categories (wine, fruits, meat, fish, sweets, gold), purchase channel counts (web/store/catalog), web visit frequency, and campaign-response flags

## Workflow

### 1. Data Cleaning
- Checked and handled missing values (rows with null `Income` dropped)
- Converted `Dt_Customer` to a proper datetime type

### 2. Feature Engineering
| Feature | Description |
|---|---|
| `Age` | Derived from `Year_Birth` |
| `Total_children` | `Kidhome` + `Teenhome` |
| `Total_spending` | Sum of spend across all 6 product categories |
| `Customer_since` | Days since enrollment (from `Dt_Customer`) |
| `AcceptedAny` | Flag for whether a customer accepted at least one of the 5 campaigns or the final offer |
| `AgeGroup` | Binned age brackets (18–29, 30–39, … 70+) for group comparisons |

### 3. Exploratory Data Analysis
- Distribution plots for age, income, and total spending
- Income by education level, and spending by marital status (boxplots)
- Correlation heatmap across income, age, recency, spending, and purchase channels
- Pivot table of average income by education × marital status
- Campaign acceptance rate by marital status
- Average income by age group

### 4. Clustering
- **Features used:** Age, Income, Total_spending, NumWebPurchases, NumStorePurchases, NumWebVisitsMonth, Recency
- **Scaling:** `StandardScaler`
- **Model selection:** Elbow method (WCSS) tested for K = 2 to 9
- **Final model:** K-Means with **K = 6** clusters
- **Visualization:** PCA reduced the scaled features to 2 components (`PCA1`, `PCA2`) for a 2D scatter plot of the clusters

### 5. Model Persistence
The trained model and scaler are saved with `joblib` for reuse without retraining:
- `kmeans_model.pkl`
- `scaler.pkl`

### 6. Interactive Dashboard
A Streamlit dashboard was built on top of the clustering output to let a user explore the segments visually — filtering by cluster and viewing each segment's profile — rather than reading the results only from static notebook plots.

## Resulting Segments

| Cluster | Size | Avg Age | Avg Income | Avg Spending | Web Purchases | Store Purchases | Web Visits/Mo | Recency | Profile |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 639 | 48 | ₹30,478 | ₹88 | 2.1 | 3.0 | 7.0 | 42.9 | Low income, low spend, low engagement |
| 1 | 271 | 58 | ₹60,109 | ₹929 | 7.5 | 8.3 | 6.2 | 73.8 | Mid-high income, high spend, very active across web & store |
| 2 | 317 | 61 | ₹58,523 | ₹763 | 6.9 | 7.4 | 6.1 | 20.9 | Mid-high income, high spend, most recently active |
| 3 | 293 | 47 | ₹79,877 | ₹1,333 | 4.5 | 8.4 | 2.5 | 47.5 | Highest income, highest spenders, store-preferring |
| 4 | 282 | 70 | ₹74,849 | ₹1,230 | 4.5 | 8.5 | 2.4 | 53.5 | Older, high income, high spend, store-preferring |
| 5 | 414 | 66 | ₹40,946 | ₹140 | 2.3 | 3.7 | 5.7 | 61.8 | Older, lower income, low spend, low engagement |

*(Values are cluster averages; see the notebook for full derivation.)*

## Tech Stack
- **Language:** Python
- **Data handling:** pandas, NumPy
- **Visualization:** matplotlib, seaborn
- **Machine Learning:** scikit-learn (StandardScaler, KMeans, PCA)
- **Model persistence:** joblib
- **Dashboard:** Streamlit

## Repository Structure
```
├── Customer_Segmentation_ML.ipynb   # Full analysis: cleaning, EDA, feature engineering, clustering
├── customer_segmentation.csv        # Raw dataset
├── segmentation.py                           # Streamlit dashboard
└── README.md
```

## How to Run
```bash
# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn joblib streamlit

# Run the notebook
jupyter notebook Customer_Segmentation_ML.ipynb

# Launch the dashboard
streamlit run app.py
```

## Key Takeaways
- Customer behaviour splits cleanly into six segments driven mainly by income, total spend, and purchase channel preference (web vs. store) rather than by age alone.
- The highest-value segments (Clusters 3 & 4) are high-income, high-spend, and prefer in-store purchasing — a useful signal for targeting premium or loyalty offers.
- The lowest-engagement segment (Cluster 0) has the lowest income and spend, suggesting a different, lower-cost marketing approach would be more effective there.
