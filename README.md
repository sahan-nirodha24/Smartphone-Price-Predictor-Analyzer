# 📱 Smartphone Price Predictor & Analyzer

A comprehensive machine learning project that predicts smartphone prices based on technical specifications and provides market analysis through an interactive web dashboard.

## 🚀 Project Overview

This project leverages machine learning algorithms to predict smartphone prices using key features such as RAM, storage, processor performance, camera specifications, and battery capacity. The system provides practical value for market analysts, e-commerce platforms, and consumers by revealing pricing patterns that aren't immediately obvious through manual comparison.

### ✨ Key Features

- **🎯 Accurate Price Predictions**: Get instant price estimates based on smartphone specifications
- **⚖️ Smart Comparisons**: Compare multiple phones side-by-side with value scoring
- **💰 Budget Planning**: Find optimal phones within your budget constraints
- **📊 Market Insights**: Understand market trends and segments
- **🌍 Multi-Currency Support**: Prices displayed in INR, USD, LKR, and EUR

## 📊 Dataset

The project uses a comprehensive dataset containing **1,836 smartphone listings from 2023**, sourced from Kaggle:

| Feature | Description |
|---------|-------------|
| Phone Name | Model names and variants |
| Rating | User ratings out of 5 |
| Number of Ratings | Total user reviews |
| RAM | Memory capacity in GB |
| ROM/Storage | Internal storage in GB |
| Camera Specifications | Front and rear camera megapixels |
| Battery | Capacity in mAh |
| Processor | Performance specifications |
| Price | Listed price in Indian Rupees |
| Date of Scraping | Data collection timestamp |

## 🤖 Machine Learning Models

The project implements and compares **four different machine learning algorithms**:

### 1. 🌲 Random Forest Regressor ⭐ (Best Model)
- **Performance**: ~85% R² score
- **Status**: Primary model used for predictions
- **Contributor**: ITBNM-2211-0181 (M P S J Rathnayaka)

### 2. 📈 Linear Regression
- **Performance**: R² Score: 0.099, MAE: 123.39, RMSE: 146.14
- **Status**: Baseline comparison model
- **Contributor**: ITBNM-2211-0129 (W S N Fernando)

### 3. 🎯 Support Vector Machine (SVM)
- **Implementation**: Alternative regression approach
- **Contributor**: ITBNM-2211-0144 (G D V Lakshan)

### 4. 🔍 K-Nearest Neighbors (KNN)
- **Implementation**: Instance-based learning model
- **Contributor**: ITBNM-2211-0132 (K G D K Gunasekara)

## 📁 Project Structure

```
Smartphone-Price-Predictor-Analyzer/
├── 🌐 app.py                                    # Streamlit web application
├── 📓 Data_Pre_Processing_With_Model.ipynb     # Main analysis notebook
├── 📄 mobile_prices_2023.csv                   # Original dataset
├── 📄 mobile_prices_cleaned.csv               # Preprocessed dataset
├── 🤖 smartphone_price_predictor.pkl          # Random Forest model
├── 🤖 smartphone_price_predictor_linear.pkl   # Linear Regression model
├── 🤖 smartphone_price_predictor_svm.pkl      # SVM model
├── 🤖 smartphone_price_predictor_knn.pkl      # KNN model
├── ⚙️ scaler.pkl                              # Feature scaler for Random Forest
├── ⚙️ scaler_linear.pkl                       # Feature scaler for Linear Regression
├── ⚙️ scaler_svm.pkl                          # Feature scaler for SVM
├── ⚙️ scaler_knn.pkl                          # Feature scaler for KNN
└── 📋 README.md                               # Project documentation
```

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8 or higher
- Git (for cloning the repository)

### Required Libraries

```python
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
streamlit>=1.0.0
plotly>=5.0.0
scikit-learn>=1.0.0
joblib>=1.0.0
```

### 📥 Installation Steps

1. **Clone the repository**
```bash
git clone <repository-url>
cd Smartphone-Price-Predictor-Analyzer
```

2. **Create a virtual environment (recommended)**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the Streamlit application**
```bash
python -m streamlit run app.py
```

5. **Open your browser** to `http://localhost:8501`

## 🌐 Web Application Features

### 1. 📱 Single Phone Price Prediction
- Interactive sliders for specification input
- Real-time price predictions
- Category classification (Budget, Mid-Range, Premium, Flagship)
- Detailed specification summary

### 2. ⚖️ Phone Comparison Tool
- Compare 2-3 smartphones simultaneously
- Side-by-side specification analysis
- Value score calculation
- Visual comparison charts

### 3. 💰 Budget Forecasting
- Set budget constraints and priorities
- Smart specification recommendations
- Budget analysis (under/over/within budget)
- Optimization suggestions based on user preferences

### 4. 📊 Market Analysis Dashboard
- Market segment distribution
- Average price by category
- Market share insights
- Consumer trend analysis

## 🔧 Data Preprocessing

The project includes comprehensive data preprocessing steps:

1. **🧹 Data Cleaning**: Handle missing values and inconsistent formats
2. **⚙️ Feature Engineering**: Extract numeric values from specification strings
3. **🏷️ Label Encoding**: Convert categorical variables to numeric format
4. **📏 Feature Scaling**: Standardize numerical features using StandardScaler
5. **📊 Price Categorization**: Classify phones into budget segments

## 📈 Model Performance

### 🌲 Random Forest (Best Model)
- **R² Score**: ~85%
- **Training Dataset**: 5,000+ smartphone listings
- **Feature Count**: 342 engineered features

### 📈 Linear Regression (Baseline)
- **R² Score**: 0.099
- **MAE**: 123.39
- **RMSE**: 146.14

## 👥 Team Contributions

### 🔧 Data Preprocessing
- **ITBNM-2211-0136** (J M P M Jayakody): Data preprocessing and feature engineering

### 🤖 Model Development
- **ITBNM-2211-0181** (M P S J Rathnayaka): Random Forest implementation
- **ITBNM-2211-0129** (W S N Fernando): Linear Regression and web dashboard
- **ITBNM-2211-0144** (G D V Lakshan): SVM implementation
- **ITBNM-2211-0132** (K G D K Gunasekara): KNN implementation

### 🎯 Evaluation and Deployment
- **ITBNM-2211-0129** (W S N Fernando): Model evaluation and Streamlit dashboard

## 💡 Usage Examples

### Price Prediction

```python
# Example smartphone specifications
phone_specs = {
    'RAM': 8,           # GB
    'Storage': 128,     # GB
    'Battery': 5000,    # mAh
    'Rear_Camera': 48,  # MP
    'Front_Camera': 13, # MP
    'Processor_Score': 8.5,
    'Rating': 4.2
}

predicted_price = model.predict([phone_specs])
print(f"Predicted Price: ₹{predicted_price[0]:,.2f}")
```

### Market Segments

The application provides insights into:

| Segment | Price Range | Market Share | Description |
|---------|-------------|--------------|-------------|
| 💚 **Budget** | ₹5K-15K | 34% | Entry-level phones |
| 💙 **Mid-Range** | ₹15K-30K | 41% | Balanced features |
| 💜 **Premium** | ₹30K-50K | 20% | High-end features |
| 🖤 **Flagship** | ₹50K+ | 5% | Cutting-edge technology |

## 🔒 Privacy & Security

- ✅ No personal data collection
- ✅ All processing runs locally
- ✅ No data stored or transmitted
- ✅ Open-source implementation

## ⚠️ Limitations

- Predictions are estimates based on historical 2023 data
- Model performance may vary with newer smartphone releases
- Market dynamics and pricing strategies may change over time
- Currency fluctuations may affect price accuracy

## 🚀 Future Enhancements

- [ ] Real-time data integration
- [ ] Advanced feature engineering
- [ ] Deep learning model implementation
- [ ] Mobile application development
- [ ] Multi-language support
- [ ] Brand-specific price analysis
- [ ] Market trend forecasting

## 🔧 Troubleshooting

### Common Issues

**Issue**: `ModuleNotFoundError`
**Solution**: Ensure all dependencies are installed: `pip install -r requirements.txt`

**Issue**: Streamlit app won't start
**Solution**: Check if port 8501 is available or use: `streamlit run app.py --server.port 8502`

**Issue**: Model files not found
**Solution**: Ensure all `.pkl` files are in the project directory

## 📄 License

This project is developed as part of an academic assignment for the **Bachelor of Information Technology (Hons) Networking and Mobile Computing** program at Horizon Campus.

**⭐ If you find this project helpful, please give it a star!**

*Made with ❤️ by the Horizon Campus BIT NMC Team*
