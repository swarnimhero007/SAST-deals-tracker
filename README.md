# 📊 SAST Deal Tracker - AI-Powered Bulk Deals Monitor

A powerful Streamlit dashboard with AI-powered stock analysis to track bulk deals for top 50 whale clients from the National Stock Exchange (NSE) of India.

## 🌟 Features

### Core Features
- **Real-time Data Fetching**: Automatically fetch bulk deals data from NSE API
- **Date Range Selection**: 
  - Quick presets: 1 Day, 1 Week, 1 Month, 1 Year
  - Custom date range picker
- **Client Filtering**: Track specific whale clients or all 50 major players
- **CSV Export**: Download filtered data for further analysis
- **Interactive Dashboard**: Clean, user-friendly interface built with Streamlit
- **Pagination**: View bulk deals data with smooth pagination (50 records per page)

### AI-Powered Analysis 🤖
- **Perplexity AI Integration**: Advanced stock analysis using Perplexity's Sonar Pro model
- **Swing Trade Recommendations**: AI-generated entry zones, targets, and stop-loss levels
- **Institutional Activity Tracking**: Identify major institutional buying patterns
- **Fundamental Analysis**: Revenue growth, profit growth, ROE, debt-to-equity ratio
- **Technical Analysis**: RSI, MACD, moving averages, support/resistance levels
- **Price History Charts**: Interactive candlestick charts with volume analysis
- **Latest News (Top 5)**: Real-time news aggregation with citations
- **Risk Assessment**: Comprehensive risk analysis for each stock
- **PDF Export**: Generate professional PDF reports with charts and analysis

### Price Charts & Technical Indicators 📈
- **OHLC Candlestick Charts**: Interactive price charts using Plotly
- **Moving Averages**: 20-day and 50-day moving averages
- **Volume Analysis**: Volume bars with color-coded buy/sell pressure
- **Yahoo Finance Integration**: Real-time price data for NSE stocks (.NS suffix)

## 📋 Top 50 Whale Clients Tracked

The dashboard monitors bulk deals from 50 major market participants including:
- **Mutual Funds**: SBI, ICICI, HDFC, Kotak, Nippon, Aditya Birla, Franklin Templeton, etc.
- **Foreign Institutional Investors**: Morgan Stanley, Goldman Sachs, BNP Paribas, Credit Suisse, etc.
- **Proprietary Trading Firms**: Graviton Research, Tower Research, Jump Trading, Optiver, etc.
- **Hedge Funds**: Jana Chetna, Copthall Mauritius, CLSA, etc.
- **Investment Firms**: Mirae Asset, Motilal Oswal, Kotak Securities, etc.

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher (recommended: Python 3.11)
- pip (Python package installer)
- Internet connection for data fetching

### Step 1: Clone or Download

Download or navigate to the project directory:
```bash
cd "c:\Users\LENOVO\Desktop\SAST Deal Tracker"
```

### Step 2: Install Dependencies

Install required Python packages:
```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install streamlit==1.29.0 pandas==2.1.4 requests==2.31.0 openai==1.12.0 pydantic==2.5.3 yfinance==0.2.33 plotly==5.18.0 reportlab==4.0.8 kaleido==0.2.1
```

## 🔑 API Configuration

### Perplexity API Key (Required for AI Analysis)

1. Get your API key from [Perplexity AI](https://www.perplexity.ai/)
2. Enter the API key in the sidebar when using the app
3. For deployment, store it in `.streamlit/secrets.toml`:

```toml
PERPLEXITY_API_KEY = "your-api-key-here"
```

## 📱 Usage

### Running the Dashboard Locally

1. Open a terminal/command prompt
2. Navigate to the project directory:
   ```bash
   cd "c:\Users\LENOVO\Desktop\SAST Deal Tracker"
   ```

3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

4. The dashboard will automatically open in your default web browser at `http://localhost:8501`

### Using the Dashboard

#### 1. **Fetch Bulk Deals Data**
   - **Select Date Range**: Choose from quick options or use custom dates
   - **Filter Clients**: Select "All" or specific whale clients
   - **Choose Data Source**: Download from NSE or upload your own CSV
   - Click **"🔍 Fetch Deals from NSE"** button
   - Data is automatically saved to `saved_data/` folder

#### 2. **View & Filter Data**
   - Browse paginated bulk deals (50 per page)
   - Use pagination controls: First, Previous, Next, Last
   - Download filtered data as CSV

#### 3. **AI Stock Analysis** 🤖
   - Enter number of stocks to analyze (1-20)
   - Add optional custom instructions (e.g., "Focus on small-cap stocks")
   - Click **"🚀 Analyze Stocks with AI"**
   - Wait 30-60 seconds for AI analysis

#### 4. **Review Analysis Results**
   - View AI-generated swing trade opportunities
   - Check institutional activity and notable buyers
   - Review fundamental metrics (Market Cap, ROE, Revenue Growth)
   - Analyze price history and charts
   - Read latest news (Top 5) with citations
   - Review technical indicators (RSI, MACD, Support/Resistance)
   - Get trade recommendations (Entry, Targets, Stop-Loss)

#### 5. **Export to PDF** 📄
   - Click **"📥 Generate PDF Report"**
   - Download comprehensive PDF with:
     - Market context
     - Individual stock analysis
     - Price charts (if kaleido is installed)
     - News articles with citations
     - Trade recommendations

## 📂 Project Structure

```
SAST Deal Tracker/
│
├── app.py                      # Main Streamlit application
├── clients_data.json           # Top 50 whale clients list
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── .streamlit/
│   ├── config.toml            # Streamlit configuration
│   └── secrets.toml           # API keys (not in git)
├── saved_data/                # Auto-saved CSV files
└── .gitignore                 # Git ignore file
```

## 🔧 Configuration

### Streamlit Configuration (`config.toml`)

```toml
[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
font = "sans serif"

[server]
maxUploadSize = 200
enableXsrfProtection = true
```

### Date Format
The NSE API expects dates in `DD-MM-YYYY` format. The dashboard automatically handles this conversion.

## 📊 Data Fields

### Bulk Deals Data
- Date
- Symbol/Stock Name
- Client Name
- Buy/Sell indicator
- Quantity Traded
- Trade Price
- Remarks

### AI Analysis Data
- Overall Score (0-100)
- Institutional Activity
- Market Cap & Sector
- Fundamental Metrics (Revenue Growth, Profit Growth, ROE, Debt/Equity)
- Technical Indicators (Current Price, 50 DMA, 200 DMA, RSI, MACD)
- Price History (High, Low, Change %, Volume)
- Latest News (Top 5 with URLs)
- Trade Recommendations (Entry Zone, Targets, Stop-Loss, Risk-Reward)
- Risk Assessment

## 🎯 AI Analysis Scoring Methodology

The AI rates stocks on a scale of 0-100 based on:
- **Institutional Interest** (25 points): Strength of institutional buying
- **Fundamental Quality** (25 points): Revenue, profit growth, ROE, debt levels
- **Technical Setup** (30 points): Price action, indicators, support/resistance
- **Risk-Reward Ratio** (20 points): Potential upside vs downside risk

## 📈 Swing Trade Definition

A swing trade is a medium-term position (3-12 weeks) that aims to capture a significant price move of 10-30%, based on:
- Strong institutional buying interest
- Favorable technical setup with clear support/resistance levels
- Positive fundamental momentum
- Risk-reward ratio of at least 1:2
- Clear entry, exit, and stop-loss levels

## ⚠️ Important Notes

1. **NSE API Availability**: The NSE API may have rate limits or temporary downtime. If you encounter errors, wait a few moments and try again.

2. **Perplexity API Costs**: AI analysis uses the Perplexity API which may have usage costs. Check your API plan limits.

3. **Internet Connection**: Required for fetching data from NSE, Yahoo Finance, and Perplexity AI.

4. **Browser Compatibility**: Best viewed in modern browsers (Chrome, Firefox, Edge, Safari).

5. **Data Accuracy**: Data is fetched from NSE, Yahoo Finance, and news sources. Always verify critical information with official sources.

6. **Disclaimer**: This tool is for educational and research purposes only. It is not financial advice. Always do your own research before making investment decisions.

7. **Kaleido for PDF Charts**: If chart export fails, reinstall kaleido:
   ```bash
   pip uninstall kaleido -y
   pip install kaleido==0.2.1
   ```

## 🐛 Troubleshooting

### Issue: "Error fetching data from NSE"
- **Solution**: Check internet connection. NSE servers may be temporarily unavailable. Wait and retry.

### Issue: "No data available for selected date range"
- **Solution**: Try a different date range. Bulk deals may not occur every day for selected clients.

### Issue: "Module not found" error
- **Solution**: Install all dependencies: `pip install -r requirements.txt`

### Issue: "Perplexity API error"
- **Solution**: 
  - Verify API key is correct
  - Check API quota/credits
  - Ensure internet connection is stable

### Issue: "Could not save chart for PDF"
- **Solution**: 
  ```bash
  pip uninstall kaleido -y
  pip install kaleido==0.2.1
  ```
  - If issue persists, PDF will be generated without charts

### Issue: "Price data not loading"
- **Solution**: 
  - Verify stock symbol exists on NSE
  - Check Yahoo Finance connectivity
  - Try a different date range

### Issue: Dashboard won't start
- **Solution**: 
  - Ensure Streamlit is installed: `pip install streamlit`
  - Verify you're in the correct directory
  - Check Python version (3.8+)

## 🚀 Deployment on Streamlit Cloud

### Step 1: Prepare Repository
```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/YOUR_USERNAME/sast-deal-tracker.git
git push -u origin main
```

### Step 2: Deploy
1. Go to [share.streamlit.io](https://share.streamlit.io/)
2. Sign in with GitHub
3. Click **"New app"**
4. Select your repository
5. Set main file: `app.py`
6. Add secrets in Advanced settings:
   ```toml
   PERPLEXITY_API_KEY = "your-api-key-here"
   ```
7. Click **"Deploy"**

Your app will be live at: `https://YOUR_USERNAME-sast-deal-tracker.streamlit.app`

## 📝 Technology Stack

- **Frontend**: Streamlit
- **Data Processing**: Pandas
- **API Requests**: Requests
- **AI Analysis**: OpenAI SDK (Perplexity API)
- **Data Validation**: Pydantic
- **Stock Data**: yfinance
- **Charts**: Plotly
- **PDF Generation**: ReportLab
- **Image Export**: Kaleido

## 🔒 Security & Privacy

- API keys are stored securely in Streamlit secrets
- No user data is collected or stored
- All analysis is performed server-side
- Data is fetched from public APIs only

## 📄 License

This project is for educational and research purposes. Please ensure compliance with:
- NSE's terms of service
- Yahoo Finance terms of use
- Perplexity AI API terms
- News sources' citation policies

## 🤝 Contributing

Contributions are welcome! Potential enhancements:
- Additional technical indicators
- Backtesting functionality
- Portfolio tracking
- Alert notifications
- Multi-timeframe analysis
- Sector analysis
- Comparison charts

## 📧 Support & Resources

- **Streamlit**: [docs.streamlit.io](https://docs.streamlit.io/)
- **NSE India**: [www.nseindia.com](https://www.nseindia.com/)
- **Perplexity AI**: [www.perplexity.ai](https://www.perplexity.ai/)
- **Yahoo Finance**: [finance.yahoo.com](https://finance.yahoo.com/)

## 🎓 Educational Disclaimer

**This tool is for educational and informational purposes only.**

- Not financial advice or investment recommendation
- Past performance does not guarantee future results
- Always conduct your own research (DYOR)
- Consult a licensed financial advisor before investing
- Trading involves risk of loss
- Use at your own risk

---

**Happy Trading! 📈🚀**

*Built with ❤️ using Streamlit, Perplexity AI, and Open Source Tools*
