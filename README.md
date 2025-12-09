# 🧠 PokeClinicX
Enhancing Recruitment Rate in Clinical Trials through Predictive Modeling
## 🚀 Overview

Clinical trial recruitment is one of the most challenging aspects of medical research due to limited outreach, complex eligibility criteria, and participant hesitation.
PokeClinicX transforms this process by forecasting recruitment success and categorizing trials into 13 Pokémon-inspired types (🔥 Fire, 💧 Water, ❄️ Ice, etc.) based on demographic and psychological patterns.
This enables improved trial matching and benchmarking.

## 📦 Repository Structure
| File / Folder                  | Description                                               |
| ------------------------------ | --------------------------------------------------------- |
| `raw_unclassified_dataset.csv` | Original dataset before processing                        |
| `classified_dataset.csv`       | Dataset after classification into Pokémon-type categories |
| `PokeClinicX_notebook.ipynb`   | Full ML pipeline implementation                           |
| `README.md`                    | Project documentation                                     |
| `images/`                      | Visuals used in reports & slides                          |


## 🧪 Approach & Methodology
### 🔍 Understanding the Problem

Clinical trials struggle with low participant recruitment due to:

Limited outreach networks

Complex eligibility filters

Lack of personalized matching

Goal: Efficiently predict recruitment success and classify trials for benchmarking.

## ⚙️ Methodology Pipeline

### External Classification of Trial Data
Categorize trials into 13 Pokémon-types based on features.

### Predictive Modeling
ML models predict recruitment success for each type.

### Data Preprocessing
Encode categorical data, scale numerical values, apply NLP to textual features.

### Training & Validation
Evaluate using metrics such as RMSE, MAE, Precision@K, Recall@K.

### Deployment & Feedback
Insights help organizers refine strategies and optimize outcomes.

## 🧠 Model Choice & Setup
### 🤖 Models Used<br>
XGBoost	High accuracy with structured trial data
BERT	Extracts contextual insights from text for trial matching
K-Means	Clusters trials/users by recruitment behavior
DBSCAN	Detects outliers in dense data distributions<br>
### 🌐 End-to-End ML Pipeline
Data Collection → External Classification → Preprocessing → Feature Engineering → Training & Evaluation → Deployment

## 📈 Model Training & Evaluation
### 📊 Evaluation Metrics

Prediction Accuracy

RMSE, MAE, R²

Trial Matching Performance

Precision@K, Recall@K, MRR (Mean Reciprocal Rank)

## 🌟 Results & Visualization

Predictive models accurately forecast recruitment success

Pokémon-type categorization highlights demographic trends

Visual aids used:

Heatmaps: attribute-recruitment correlation

Type comparison: bar & pie charts for distributions

Geographical maps for regional performance

## ⏭️ Future Enhancements

Interactive app for real-time user engagement

Psychological interview and type assignment

Reinforcement learning-based reward optimization

Trial recommendation system with incentives

## 🛠 Tools & Libraries
Category	Technologies<br>
Machine Learning	Scikit-learn, XGBoost, LightGBM
Deep Learning / NLP	TensorFlow / Keras / PyTorch, BERT, HuggingFace
Visualization & Data Processing	Pandas, NumPy, Matplotlib, Seaborn
## 👨‍💻 Author

Aayush Sharma<br>
Computer Science & Engineering, Chandigarh University
Passionate about AI, clinical healthcare innovation & research

## 🤝 Contributions

Contributions, issues, and feature requests are welcome!
Feel free to star ⭐ the project if you found it interesting.

## 📜 License

MIT License – Free to use, modify, and distribute.