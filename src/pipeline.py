import os, json, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, classification_report

BASE=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW=os.path.join(BASE,"data","raw")
PROC=os.path.join(BASE,"data","processed")
MODELS=os.path.join(BASE,"models")
REPORTS=os.path.join(BASE,"reports")
os.makedirs(PROC,exist_ok=True); os.makedirs(MODELS,exist_ok=True); os.makedirs(REPORTS,exist_ok=True)

files={"Nykaa":"nykaa_campaign_data_with_nulls.csv","Purplle":"purplle_campaign_data_with_nulls.csv","Tira":"tira_campaign_data_with_nulls.csv"}

def load_data():
    frames=[]
    for brand, fn in files.items():
        p=os.path.join(RAW,fn)
        df=pd.read_csv(p)
        df["Brand"]=brand
        frames.append(df)
    return pd.concat(frames,ignore_index=True)

def clean_and_engineer(df):
    df=df.copy()
    df.columns=[c.strip() for c in df.columns]
    df=df.drop_duplicates()
    numeric=["Duration","Impressions","Clicks","Leads","Conversions","Revenue","Acquisition_Cost","ROI","Engagement_Score"]
    categorical=["Campaign_Type","Target_Audience","Channel_Used","Language","Customer_Segment","Brand"]
    for c in numeric: df[c]=pd.to_numeric(df[c],errors="coerce")
    for c in categorical: df[c]=df[c].astype("string").str.strip()
    df["Date"]=pd.to_datetime(df["Date"],errors="coerce",dayfirst=True)
    for c in numeric: df[c]=df[c].fillna(df[c].median())
    for c in categorical: df[c]=df[c].fillna("Unknown")
    df["Date"]=df["Date"].fillna(df["Date"].median())
    df["Calculated_ROI"]=np.where(df["Acquisition_Cost"]>0, df["Revenue"]/df["Acquisition_Cost"], np.nan)
    df["Profit_Flag"]=(df["ROI"]>=1.0).astype(int)
    df["Profit_Loss"]=df["Profit_Flag"].map({1:"Profit",0:"Loss"})
    df["CTR"]=np.where(df["Impressions"]>0,df["Clicks"]/df["Impressions"],0)
    df["Lead_Rate"]=np.where(df["Clicks"]>0,df["Leads"]/df["Clicks"],0)
    df["Conversion_Rate"]=np.where(df["Leads"]>0,df["Conversions"]/df["Leads"],0)
    df["Revenue_per_Conversion"]=np.where(df["Conversions"]>0,df["Revenue"]/df["Conversions"],0)
    df["Month"]=df["Date"].dt.month
    df["Year"]=df["Date"].dt.year
    df["Channel_Count"]=df["Channel_Used"].str.split(",").str.len()
    return df

def build_models(df):

    rev_features=["Brand","Campaign_Type","Target_Audience","Duration","Channel_Used","Impressions","Clicks","Leads","Conversions","Acquisition_Cost","Language","Engagement_Score","Customer_Segment","Month","Year","Channel_Count"]
    X=df[rev_features]; y=df["Revenue"]
    cat=[c for c in rev_features if X[c].dtype=="object" or str(X[c].dtype)=="string"]
    num=[c for c in rev_features if c not in cat]
    prep=ColumnTransformer([("num",Pipeline([("imp",SimpleImputer(strategy="median")),("scale",StandardScaler())]),num),
                           ("cat",Pipeline([("imp",SimpleImputer(strategy="most_frequent")),("oh",OneHotEncoder(handle_unknown="ignore"))]),cat)])
    rev=Pipeline([("prep",prep),("model",RandomForestRegressor(n_estimators=120,max_depth=18,random_state=42,n_jobs=-1))])
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42)
    rev.fit(Xtr,ytr); pred=rev.predict(Xte)
    rev_metrics={"MAE":float(mean_absolute_error(yte,pred)),"RMSE":float(mean_squared_error(yte,pred)**.5),"R2":float(r2_score(yte,pred))}
    joblib.dump(rev,os.path.join(MODELS,"revenue_model.pkl"))

    clf_features=["Brand","Campaign_Type","Target_Audience","Duration","Channel_Used","Impressions","Clicks","Leads","Conversions","Revenue","Acquisition_Cost","Language","Engagement_Score","Customer_Segment","Month","Year","Channel_Count"]
    X=df[clf_features]; y=df["Profit_Flag"]
    cat=[c for c in clf_features if X[c].dtype=="object" or str(X[c].dtype)=="string"]; num=[c for c in clf_features if c not in cat]
    prep2=ColumnTransformer([("num",Pipeline([("imp",SimpleImputer(strategy="median")),("scale",StandardScaler())]),num),
                           ("cat",Pipeline([("imp",SimpleImputer(strategy="most_frequent")),("oh",OneHotEncoder(handle_unknown="ignore"))]),cat)])
    clf=Pipeline([("prep",prep2),("model",RandomForestClassifier(n_estimators=120,max_depth=18,class_weight="balanced",random_state=42,n_jobs=-1))])
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
    clf.fit(Xtr,ytr); cp=clf.predict(Xte)
    clf_metrics={"Accuracy":float(accuracy_score(yte,cp)),"Precision":float(precision_score(yte,cp,zero_division=0)),"Recall":float(recall_score(yte,cp,zero_division=0)),"F1":float(f1_score(yte,cp,zero_division=0))}
    joblib.dump(clf,os.path.join(MODELS,"profit_loss_model.pkl"))
    with open(os.path.join(REPORTS,"metrics.json"),"w") as f: json.dump({"revenue_regression":rev_metrics,"profit_loss_classification":clf_metrics},f,indent=2)
    return rev_metrics,clf_metrics

def main():
    raw=load_data()
    df=clean_and_engineer(raw)
    df.to_csv(os.path.join(PROC,"combined_cleaned_campaign_data.csv"),index=False)
    rev,clf=build_models(df)
    summary=df.groupby("Brand").agg(Campaigns=("Campaign_ID","count"),Revenue=("Revenue","sum"),Avg_ROI=("ROI","mean"),Avg_Conversions=("Conversions","mean")).reset_index()
    summary.to_csv(os.path.join(REPORTS,"brand_summary.csv"),index=False)
    print("Completed successfully")
    print("Rows:",len(df)); print("Revenue metrics:",rev); print("Classification metrics:",clf)
if __name__=="__main__": main()
