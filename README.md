# ISIC 2024 - Skin Cancer Detection with 3D-TBP

## Competition Overview
This repository contains code for the **ISIC 2024 - Skin Cancer Detection with 3D-TBP** competition. The competition dataset can be accessed at the following link:
[ISIC 2024 Challenge - Kaggle](https://www.kaggle.com/competitions/isic-2024-challenge)

## Description
Skin cancer can be deadly if not detected early, but many populations lack specialized dermatologic care. Over the past several years, **dermoscopy-based AI algorithms** have been shown to assist clinicians in diagnosing melanoma, basal cell carcinoma, and squamous cell carcinoma. However, identifying which individuals should see a clinician in the first place can have a significant impact on early diagnosis and disease prognosis.

This competition focuses on **triaging applications** that can help underserved populations and improve early skin cancer detection. The dataset leverages **3D TBP technology**, providing lesion images from thousands of patients across three continents, resembling real-world smartphone photographs. The challenge is to develop AI models that accurately differentiate between **histologically-confirmed malignant and benign lesions**.

## Evaluation
### **Primary Scoring Metric**
Submissions are evaluated using **Partial Area Under the ROC Curve (pAUC)** above **80% True Positive Rate (TPR)** for binary classification of malignant examples. The implementation can be found in the official competition notebook (**ISIC pAUC-aboveTPR**).

The **Receiver Operating Characteristic (ROC) curve** illustrates the diagnostic ability of a classifier as its discrimination threshold varies. However, in clinical practice, low TPR values are unacceptable. Cancer detection systems require high sensitivity, so this metric prioritizes the **area under the ROC curve above 80% TPR**. Scores range from **0.0 to 0.2**.

The following example illustrates two arbitrary algorithms (**Ca** and **Cb**) with shaded regions representing their pAUC values at a specified minimum TPR:

![pAUC Example](https://upload.wikimedia.org/wikipedia/commons/5/5f/Roc_curve.svg)

## Repository Structure
```
├── data/                      # Dataset directory (not included in repository)
├── notebooks/                 # Jupyter notebooks for EDA and model training
├── models/                    # Trained model weights
├── src/                       # Source code for data processing and training
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
```

## Getting Started
### **1. Clone the repository**
```sh
git clone https://github.com/HiTranh2504/group-recommender-systems-for-movielens-100K-dataset.git
cd ISIC-Skin-Cancer-Github
```

### **2. Install dependencies**
```sh
pip install -r requirements.txt
```

### **3. Download the dataset**
- Visit [ISIC 2024 Challenge](https://www.kaggle.com/competitions/isic-2024-challenge) and download the dataset.
- Extract and place it in the `data/` directory.

### **4. Run the training script**
```sh
python src/train.py
```

## Contributing
Contributions are welcome! Feel free to open issues or submit pull requests.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
