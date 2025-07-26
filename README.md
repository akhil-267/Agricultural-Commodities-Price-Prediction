# Agricultural Commodities Price Prediction

This repository contains a simple Flask web application for forecasting agricultural commodity prices and trading volumes using a RandomForest model.  The project includes a small dataset of daily prices and provides a web interface to visualize predictions with confidence intervals.

## Features

- Predict future closing prices for a selected commodity.
- Check total trading volume for each commodity.
- Visualization of forecasts with 95% confidence intervals.
- User friendly HTML/CSS interface.

## Repository Structure

```
A1_TBP_054,036,005/
├── A1_TBP_Source_Code&Datasets/
│   ├── all_agricultural_products_data.csv  # price and volume data
│   ├── main.py                             # Flask application
│   ├── model.ipynb                         # Jupyter notebook for model training
│   ├── static/
│   │   └── style.css                      # page styling
│   └── templates/                         # HTML templates
│       ├── index.html
│       ├── predict_prices.html
│       ├── prediction_result.html
│       └── volumes_traded.html
└── README.md
```

## Dataset

`all_agricultural_products_data.csv` contains daily open, high, low, close and volume values for a variety of commodities.  The Flask app uses this data to train a model on demand and generate short term forecasts.

## Getting Started

### Requirements

- Python 3
- Flask
- pandas
- numpy
- scikit-learn
- matplotlib

Install dependencies with:

```bash
pip install flask pandas numpy scikit-learn matplotlib
```

### Running the Application

```
cd A1_TBP_054,036,005/A1_TBP_Source_Code&Datasets
python main.py
```

Once running, open `http://localhost:5000` in your browser to select between price prediction or volume checks.  Enter the desired commodity and forecast horizon to view predicted prices along with confidence intervals.

## Notebook

The `model.ipynb` notebook illustrates dataset exploration and model training outside of the web interface.  It can be used to experiment with different approaches.

## License

This project is provided for educational purposes and does not include any specific license.
