# Multi-Brand Marketing Campaign Performance Analysis

## Run in VS Code

```bash
cd Multi_Brand_Marketing_Analysis
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python src/pipeline.py
streamlit run app.py
```

The pipeline combines the Nykaa, Purplle and Tira campaign CSV files, cleans missing values, creates features, trains a revenue regression model and a profit/loss classification model, and saves outputs under `data/processed`, `models`, and `reports`.

## Models
- Revenue prediction: Random Forest Regressor
- Profit/Loss prediction: Random Forest Classifier
- Profit is defined as ROI >= 1.0
- ROI is excluded from classification features to reduce leakage.
"# Multi_Brand_Marketing_Analysis_Project" 
