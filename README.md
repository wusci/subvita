# Type 2 Diabetes Risk Prediction (NHANES)

This project builds a calibrated risk model for Type 2 Diabetes using NHANES data.

## Pipeline stages
0. Project setup
1. Download NHANES XPT files
2. Extract XPT → Parquet
3. Standardize variables across cycles
4. Merge tables by SEQN
5. Feature engineering + label creation
6. Train/validation/test split

## Model A
- Normal / Prediabetes / Diabetes classification