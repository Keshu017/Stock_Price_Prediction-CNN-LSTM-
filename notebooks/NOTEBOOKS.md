# Jupyter Notebooks & Google Colab Projects

This directory contains all Jupyter notebooks used in the Tesla Stock Price Prediction project. The notebooks are hosted on Google Colab for easy access and execution without local setup.

## Available Notebooks

### 1. Hybrid CNN-LSTM Advanced Model

**Advanced Deep Learning Model for Time Series Forecasting**

- **Open in Google Colab**: [Hybrid CNN-LSTM Advanced Notebook](https://colab.research.google.com/drive/1TY4wW7Y9anmZWwGvyfrtFLeLZmcC9qYw)
- **Status**: ✅ Complete & Production Ready
- **Complexity**: Advanced
- **Training Time**: ~12 minutes on Colab GPU
- **Model Params**: 500,801

#### Features:
- Multi-scale CNN branches (3 parallel convolutions)
- Bidirectional LSTM layers for temporal context
- Early stopping & learning rate scheduling
- 6+ comprehensive visualizations
- Full performance metrics (RMSE, MAE, R², MAPE)

#### Best For:
- Advanced practitioners
- Interview preparation
- Production deployment
- Resume projects

---

### 2. Simple LSTM Beginner-Friendly Model

**Beginner-Friendly Time Series Prediction**

- **Open in Google Colab**: [Simple LSTM Beginner Notebook](https://colab.research.google.com/drive/1EOKU5ZM6eCt0QxwZQC4iEzAiTVp5fewR)
- **Status**: ✅ Complete & Working
- **Complexity**: Beginner
- **Training Time**: ~2-3 minutes on Colab
- **Model Params**: 11,701

#### Features:
- Single LSTM layer for easy understanding
- Simple dense layers
- Clear, commented code
- Quick training for experimentation
- Basic visualization

#### Best For:
- Learning LSTM concepts
- Starting with deep learning
- Quick prototyping
- Understanding time series basics

---

## How to Use

### Option 1: Google Colab (Recommended)
1. Click on the notebook link above
2. Click "Copy to Drive" to get your own copy
3. Run all cells sequentially (Ctrl+F9 or Runtime > Run all)
4. Download results if needed

### Option 2: Local Jupyter Notebook
1. Install requirements: `pip install -r requirements.txt`
2. Download the .ipynb file from Colab (File > Download > .ipynb)
3. Open in Jupyter: `jupyter notebook notebook_name.ipynb`
4. Run cells sequentially

---
## Data Source

- **Source**: Yahoo Finance (yfinance)
- **Stock**: Tesla (TSLA)
- **Period**: 2015-01-01 to 2021-06-28
- **Records**: 1,608 trading days
- **Feature**: Daily closing price

---

## Output Examples

Both notebooks generate:
- ✅ Actual vs Predicted price plots
- ✅ Error distribution analysis
- ✅ Training history curves
- ✅ Performance metrics
- ✅ Residual analysis
- ✅ Model architecture details

---

## Model Comparison

| Feature | Hybrid CNN-LSTM | Simple LSTM |
|---------|-----------------|-------------|
| Params | 500,801 | 11,701 |
| Complexity | Advanced | Beginner |
| Training Time | 12 min | 2-3 min |
| Accuracy | High | Good |
| Resume Impact | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |

---

## Troubleshooting

### Google Colab Issues
- **Out of Memory**: Reduce batch size in Cell 4
- **Slow Training**: Use GPU runtime (Runtime > Change runtime type > GPU)
- **Data Download Error**: Manually download from Yahoo Finance

### Local Jupyter Issues
- **Import Errors**: Run `pip install -r requirements.txt`
- **GPU Not Found**: Install tensorflow-gpu separately
- **Path Errors**: Adjust data paths in notebooks

---

## Next Steps

1. **Explore Both Models**: Start with Simple, then try Advanced
2. **Modify & Experiment**: Change hyperparameters, try different stocks
3. **Add Features**: Implement Volume, RSI, or other indicators
4. **Deploy**: Convert to REST API using Flask/FastAPI
5. **Improve**: Add attention mechanisms or ensemble methods

---

## License

MIT License - See [LICENSE](../LICENSE) file

---

**Last Updated**: December 18, 2025
**Status**: Ready for Production & Learning
