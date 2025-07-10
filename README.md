# Social Media Hate Speech Detection (NLP & ML)

This repository contains ML experiments focused on detecting hate speech in social media platforms, specifically YouTube and Instagram. The project explores various  approaches, including feature engineering, topic modeling, and fine-tuned transformer models.

## Project Goals
- Detect hate speech and toxic content in short-form user-generated text
- Compare traditional models (SVM, Logistic Regression, KNN) with transformer-based methods (BERT)
- Use topic modeling (LDA, LSA) to understand latent structure in online discourse

## Files Content 

| Notebook | Description |
|----------|-------------|
| `AllFeatures.ipynb` | Consolidated feature extraction across multiple methods |
| `Bert_Insta.ipynb` | Fine-tuned BERT model for Instagram hate speech detection |
| `Bert_Youtube.ipynb` | Fine-tuned BERT model for YouTube hate speech detection |
| `Instagram_Youtube_LDA.ipynb` | Topic modeling using Latent Dirichlet Allocation |
| `LSA_instagram.ipynb` | Latent Semantic Analysis on Instagram dataset |
| `LSA_youtube.ipynb` | Latent Semantic Analysis on YouTube dataset |

## Tools & Libraries
- Python
- Hugging Face Transformers
- scikit-learn
- NLTK
- Pandas, NumPy, Matplotlib

## Models
- BERT (for sequence classification)
- Logistic Regression
- SVM, KNN
- LDA, LSA (topic modeling)
