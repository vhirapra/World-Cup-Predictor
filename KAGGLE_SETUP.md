# Kaggle API Setup Guide

To use the dynamic Kaggle dataset integration, follow these steps:

## 1. Install Dependencies
```bash
pip install -r requirements.txt
```

## 2. Get Your Kaggle API Token
1. Go to https://www.kaggle.com/settings/account
2. Click "Create New API Token"
3. This downloads `kaggle.json` file

## 3. Set Up Kaggle Credentials
Place the `kaggle.json` file in one of these locations:

**Windows:**
```
C:\Users\<YourUsername>\.kaggle\kaggle.json
```

**macOS/Linux:**
```
~/.kaggle/kaggle.json
```

Then set proper permissions:
```bash
chmod 600 ~/.kaggle/kaggle.json  # macOS/Linux only
```

## 4. Verify Setup
Run your main pipeline - it will automatically download the latest dataset:
```bash
python main.py
```

## How It Works
- Every time you run `main.py`, it will download the latest international football matches from Kaggle
- Data is stored in `data/all_matches/results.csv` (overwrites previous version)
- The dataset includes all matches from 1872 onwards
- Combined with historical World Cup data for training

## Optional: Use Local File Instead
If you want to use a local file instead of Kaggle API for a specific run:
```python
# In main.py, modify the pipeline call:
combined_data = dp.combine_match_datasets(use_kaggle=False, start_year=1990)
```

## Troubleshooting
- **Authentication Error**: Ensure kaggle.json is in the correct location
- **Dataset Not Found**: Verify Kaggle API credentials are valid
- **Permission Denied**: Check file permissions on kaggle.json
