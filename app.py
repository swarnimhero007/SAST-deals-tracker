import streamlit as st
import pandas as pd
import requests
import json
import os
from datetime import datetime, timedelta
from io import StringIO, BytesIO
from openai import OpenAI
from pydantic import BaseModel
from typing import List
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak, Image
from reportlab.lib.enums import TA_CENTER, TA_LEFT

# Page configuration
st.set_page_config(
    page_title="SAST Deal Tracker",
    page_icon="📊",
    layout="wide"
)

# Define Pydantic schema for stock analysis
class TradeRecommendation(BaseModel):
    entry_zone_min: float
    entry_zone_max: float
    target_1_price: float
    target_1_gain_percent: float
    target_1_timeframe: str
    target_2_price: float
    target_2_gain_percent: float
    target_2_timeframe: str
    stop_loss: float
    stop_loss_risk_percent: float
    risk_reward_ratio: str
    position_size_percent: float
    time_horizon_weeks: int

class NewsItem(BaseModel):
    title: str
    date: str
    source: str
    url: str
    summary: str

class PriceHistory(BaseModel):
    period_high: float
    period_low: float
    period_start: float
    period_end: float
    price_change_percent: float
    average_volume: float

class StockAnalysis(BaseModel):
    rank: int
    symbol: str
    company_name: str
    overall_score: int
    institutional_activity: str
    total_buying_cr: float
    notable_buyers: List[str]
    market_cap_cr: float
    sector: str
    qoq_revenue_growth: float
    qoq_profit_growth: float
    roe_percent: float
    debt_equity_ratio: float
    key_catalyst: str
    current_price: float
    dma_50: float
    dma_200: float
    rsi: float
    macd_signal: str
    support_level: float
    resistance_level: float
    volume_trend: str
    price_history: PriceHistory
    latest_news: List[NewsItem]
    trade_recommendation: TradeRecommendation
    risks: List[str]
    trade_rationale: str

class StockAnalysisReport(BaseModel):
    top_stocks: List[StockAnalysis]
    excluded_stocks: List[dict]
    market_context: dict

# Load top 50 clients
@st.cache_data
def load_clients():
    with open('clients_data.json', 'r') as f:
        data = json.load(f)
    return data['clients']

# Fetch bulk deals CSV from NSE
def fetch_bulk_deals(start_date, end_date):
    """
    Download bulk deals CSV file from NSE
    """
    url = f"https://nseindia.com/api/historicalOR/bulk-block-short-deals?optionType=bulk_deals&from={start_date}&to={end_date}&csv=true"
    
    # NSE requires specific headers to prevent 403 errors
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Sec-Fetch-Dest': 'document',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-Site': 'none',
        'Cache-Control': 'max-age=0',
    }
    
    try:
        # Create a session to handle cookies
        session = requests.Session()
        
        # First, visit the main page to get cookies
        session.get('https://www.nseindia.com/', headers=headers, timeout=10)
        
        # Now download the CSV file
        csv_url = f"https://nseindia.com/api/historicalOR/bulk-block-short-deals?optionType=bulk_deals&from={start_date}&to={end_date}&csv=true"
        response = session.get(csv_url, headers=headers, timeout=30)
        response.raise_for_status()
        
        # Parse CSV data directly from response content
        if response.text and len(response.text) > 0:
            csv_data = StringIO(response.text)
            df = pd.read_csv(csv_data)
            return df, None
        else:
            return None, "No data returned from NSE"
            
    except requests.exceptions.RequestException as e:
        return None, f"Error downloading CSV: {str(e)}"
    except Exception as e:
        return None, f"Error parsing CSV: {str(e)}"

# Calculate date ranges
def get_date_range(option):
    """
    Calculate start and end dates based on selected option
    """
    today = datetime.now()
    
    if option == "1 Day":
        start_date = today - timedelta(days=1)
        end_date = today
    elif option == "1 Week":
        start_date = today - timedelta(weeks=1)
        end_date = today
    elif option == "1 Month":
        start_date = today - timedelta(days=30)
        end_date = today
    elif option == "1 Year":
        start_date = today - timedelta(days=365)
        end_date = today
    else:  # Custom
        return None, None
    
    return start_date.strftime("%d-%m-%Y"), end_date.strftime("%d-%m-%Y")

# Fetch stock price data from yfinance
def fetch_stock_price_data(symbol, start_date_str, end_date_str):
    """
    Fetch OHLC price data from yfinance for the given symbol and date range
    
    Args:
        symbol: Stock symbol (e.g., 'RELIANCE')
        start_date_str: Start date in format 'DD-MM-YYYY'
        end_date_str: End date in format 'DD-MM-YYYY'
    
    Returns:
        Tuple of (DataFrame, error_message)
    """
    try:
        # Convert date strings to datetime objects
        start_date = datetime.strptime(start_date_str, "%d-%m-%Y")
        end_date = datetime.strptime(end_date_str, "%d-%m-%Y")

        print(start_date)
        print(end_date)

        # Add .NS suffix for NSE stocks
        ticker = f"{symbol}.NS"
        
        # Create ticker object
        stock = yf.Ticker(ticker)

        # Download historical data
        df = stock.history(start=start_date, end=end_date)
        
        # Check if data is empty
        if df.empty:
            return None, f"No price data found for {symbol} in the specified date range"
        
        # Reset index to make date a column
        df.reset_index(inplace=True)
        
        return df, None
        
    except ValueError as ve:
        return None, f"Invalid date format: {str(ve)}"
    except Exception as e:
        return None, f"Error fetching price data for {symbol}: {str(e)}"

# Create price chart using plotly
def create_price_chart(df, symbol):
    """
    Create an interactive candlestick chart with volume
    
    Args:
        df: DataFrame with OHLC data from yfinance
        symbol: Stock symbol for chart title
    
    Returns:
        Plotly figure object
    """
    try:
        # Create subplots: 2 rows, 1 column
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.7, 0.3],
            subplot_titles=(f'{symbol} Price Chart', 'Volume')
        )
        
        # Add candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=df['Date'],
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name='OHLC',
                increasing_line_color='green',
                decreasing_line_color='red'
            ),
            row=1, col=1
        )
        
        # Add 20-day moving average
        if len(df) >= 20:
            df['MA20'] = df['Close'].rolling(window=20).mean()
            fig.add_trace(
                go.Scatter(
                    x=df['Date'],
                    y=df['MA20'],
                    name='20 MA',
                    line=dict(color='blue', width=1)
                ),
                row=1, col=1
            )
        
        # Add 50-day moving average
        if len(df) >= 50:
            df['MA50'] = df['Close'].rolling(window=50).mean()
            fig.add_trace(
                go.Scatter(
                    x=df['Date'],
                    y=df['MA50'],
                    name='50 MA',
                    line=dict(color='orange', width=1)
                ),
                row=1, col=1
            )
        
        # Volume bar chart with colors
        colors_volume = ['red' if df['Close'].iloc[i] < df['Open'].iloc[i] else 'green' 
                         for i in range(len(df))]
        
        fig.add_trace(
            go.Bar(
                x=df['Date'],
                y=df['Volume'],
                name='Volume',
                marker_color=colors_volume,
                showlegend=False
            ),
            row=2, col=1
        )
        
        # Update layout
        fig.update_layout(
            title=f'{symbol} Stock Price Analysis',
            yaxis_title='Price (₹)',
            yaxis2_title='Volume',
            xaxis2_title='Date',
            height=600,
            hovermode='x unified',
            xaxis_rangeslider_visible=False
        )
        
        # Update x-axis for both subplots
        fig.update_xaxes(rangeslider_visible=False, row=1, col=1)
        fig.update_xaxes(rangeslider_visible=False, row=2, col=1)
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating chart: {str(e)}")
        return None

# ...existing code...

# Analyze stocks using Perplexity API
def analyze_stocks_with_perplexity(df, num_stocks, additional_instructions, api_key, start_date, end_date):
    """
    Use Perplexity AI to analyze stocks from bulk deal data
    """
    try:
        # Prepare the deals data for the prompt
        deals_data = df.to_string(index=False)
        
        # Build the prompt
        prompt = f"""You are an expert stock market analyst specializing in institutional trading patterns and swing trading strategies. Analyze the following NSE block and bulk deal data to identify stocks with the highest swing trade potential.

**SWING TRADE DEFINITION:**
A swing trade is a medium-term position (3-12 weeks) that aims to capture a significant price move of 10-30%, based on:
- Strong institutional buying interest
- Favorable technical setup with clear support/resistance levels
- Positive fundamental momentum
- Risk-reward ratio of at least 1:2
- Clear entry, exit, and stop-loss levels

**BLOCK/BULK DEAL DATA:**
{deals_data}

**ANALYSIS PERIOD:** From {start_date} to {end_date}

**ANALYSIS REQUIREMENTS:**

1. **Institutional Buyer Identification:**
   - Identify which deals involve major institutional buyers (FIIs, DIIs, mutual funds, prominent trading houses like Motilal Oswal, Graviton, Goldman Sachs, Morgan Stanley, etc.)
   - Flag stocks where institutions are ACCUMULATING (buying, not selling)
   - Highlight stocks with multiple institutional buy orders in the last 30 days

2. **Fundamental Screening:**
   For each stock with institutional buying, evaluate:
   - Market capitalization and liquidity
   - Recent quarterly results (revenue & profit growth)
   - Debt-to-equity ratio (prefer < 1)
   - ROE and ROCE (prefer > 15%)
   - Sector outlook and industry tailwinds
   - Any recent corporate actions (bonus, split, buyback)

3. **Technical Analysis:**
   - Current price vs 50 DMA and 200 DMA
   - RSI level (ideal range: 40-65 for entry)
   - MACD signal (bullish crossover?)
   - Volume trend (increasing or decreasing?)
   - Key support and resistance levels
   - Recent price action (consolidation, breakout, or trending?)

4. **Price History Analysis:**
   For each stock, provide:
   - Period high and low prices
   - Start and end prices for the analysis period
   - Percentage price change
   - Average trading volume

5. **Latest News (Top 5):**
   For each stock, find and cite the 5 most recent and relevant news articles:
   - Include article title, date, source, URL, and brief summary
   - Focus on news that could impact stock performance
   - Cite credible sources (Economic Times, Business Standard, Moneycontrol, etc.)

6. **Risk Assessment:**
   - Recent volatility (Beta)
   - Drawdown from 52-week high
   - Sector-specific risks
   - Market correlation

7. **Swing Trade Potential Scoring:**
   Rate each stock on a scale of 0-100 based on:
   - Institutional interest strength (25 points)
   - Fundamental quality (25 points)
   - Technical setup favorability (30 points)
   - Risk-reward ratio (20 points)

**IMPORTANT:** Analyze and return data for TOP {num_stocks} STOCKS with the highest swing trade potential.

**ADDITIONAL INSTRUCTIONS FROM USER:**
{additional_instructions if additional_instructions else "None"}

**DATA SOURCES TO CITE:**
Please use data from:
- NSE/BSE official data
- Screener.in or Moneycontrol for fundamentals
- TradingView or Investing.com for technical data
- Recent news from Economic Times, Business Standard, MoneyControl, Livemint, etc.
- Use Yahoo Finance for price data (ticker format: SYMBOL.NS)

Provide detailed analysis with specific numbers and actionable insights for each stock."""

        # Initialize Perplexity client using OpenAI-compatible API
        # Simplified initialization to avoid proxy/httpx issues
        try:
            client = OpenAI(
                api_key=api_key,
                base_url="https://api.perplexity.ai"
            )
        except TypeError:
            # Fallback for older OpenAI SDK versions
            import httpx
            client = OpenAI(
                api_key=api_key,
                base_url="https://api.perplexity.ai",
                http_client=httpx.Client(timeout=60.0)
            )
        
        # Make API call
        completion = client.chat.completions.create(
            model="sonar-pro",
            messages=[
                {"role": "user", "content": prompt}
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "stock_analysis",
                    "strict": True,
                    "schema": StockAnalysisReport.model_json_schema()
                }
            },
            temperature=0.3
        )
        
        # Parse response
        analysis_data = StockAnalysisReport.model_validate_json(
            completion.choices[0].message.content
        )
        
        return analysis_data, None
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return None, f"Error analyzing stocks: {str(e)}\n\nDetails:\n{error_details}"

# ...existing code...

# Generate PDF report
def generate_pdf_report(analysis_data, start_date, end_date, charts_data):
    """
    Generate a comprehensive PDF report with all analysis data and charts
    """
    try:
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=30, leftMargin=30, topMargin=30, bottomMargin=18)
        
        # Container for the 'Flowable' objects
        elements = []
        
        # Define styles
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1f77b4'),
            spaceAfter=30,
            alignment=TA_CENTER
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#2ca02c'),
            spaceAfter=12,
            spaceBefore=12
        )
        
        subheading_style = ParagraphStyle(
            'CustomSubHeading',
            parent=styles['Heading3'],
            fontSize=12,
            textColor=colors.HexColor('#d62728'),
            spaceAfter=8
        )
        
        # Title
        elements.append(Paragraph("SAST Deal Tracker - AI Stock Analysis Report", title_style))
        elements.append(Paragraph(f"Analysis Period: {start_date} to {end_date}", styles['Normal']))
        elements.append(Paragraph(f"Generated on: {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}", styles['Normal']))
        elements.append(Spacer(1, 20))
        
        # Market Context
        elements.append(Paragraph("Market Context", heading_style))
        market_data = []
        for key, value in analysis_data.market_context.items():
            market_data.append([key.replace('_', ' ').title(), str(value)])
        
        market_table = Table(market_data, colWidths=[3*inch, 4*inch])
        market_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.beige),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        elements.append(market_table)
        elements.append(PageBreak())
        
        # Individual stock analysis
        for idx, stock in enumerate(analysis_data.top_stocks):
            # Stock header
            elements.append(Paragraph(f"Rank #{stock.rank}: {stock.symbol} - {stock.company_name}", title_style))
            elements.append(Paragraph(f"Overall Score: {stock.overall_score}/100", styles['Normal']))
            elements.append(Spacer(1, 12))
            
            # Institutional Activity
            elements.append(Paragraph("Institutional Activity", heading_style))
            elements.append(Paragraph(stock.institutional_activity, styles['Normal']))
            elements.append(Paragraph(f"Total Buying: Rs {stock.total_buying_cr} Cr", styles['Normal']))
            elements.append(Paragraph("Notable Buyers:", subheading_style))
            for buyer in stock.notable_buyers:
                elements.append(Paragraph(f"• {buyer}", styles['Normal']))
            elements.append(Spacer(1, 12))
            
            # Fundamentals
            elements.append(Paragraph("Fundamental Analysis", heading_style))
            fundamental_data = [
                ['Metric', 'Value'],
                ['Market Cap', f"Rs {stock.market_cap_cr} Cr"],
                ['Sector', stock.sector],
                ['Revenue Growth (QoQ)', f"{stock.qoq_revenue_growth}%"],
                ['Profit Growth (QoQ)', f"{stock.qoq_profit_growth}%"],
                ['ROE', f"{stock.roe_percent}%"],
                ['Debt/Equity', str(stock.debt_equity_ratio)],
            ]
            
            fundamental_table = Table(fundamental_data, colWidths=[3*inch, 3*inch])
            fundamental_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            elements.append(fundamental_table)
            elements.append(Paragraph(f"Key Catalyst: {stock.key_catalyst}", styles['Normal']))
            elements.append(Spacer(1, 12))
            
            # Price History
            elements.append(Paragraph("Price History", heading_style))
            price_data = [
                ['Metric', 'Value'],
                ['Period High', f"Rs {stock.price_history.period_high}"],
                ['Period Low', f"Rs {stock.price_history.period_low}"],
                ['Start Price', f"Rs {stock.price_history.period_start}"],
                ['End Price', f"Rs {stock.price_history.period_end}"],
                ['Price Change', f"{stock.price_history.price_change_percent}%"],
                ['Avg Volume', f"{stock.price_history.average_volume:,.0f}"],
            ]
            
            price_table = Table(price_data, colWidths=[3*inch, 3*inch])
            price_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            elements.append(price_table)
            elements.append(Spacer(1, 12))
            
            # Add price chart if available
            if stock.symbol in charts_data:
                try:
                    chart_img = charts_data[stock.symbol]
                    img = Image(chart_img, width=6*inch, height=4*inch)
                    elements.append(img)
                    elements.append(Spacer(1, 12))
                except:
                    pass
            
            # Technical Setup
            elements.append(Paragraph("Technical Analysis", heading_style))
            technical_data = [
                ['Metric', 'Value'],
                ['Current Price', f"Rs {stock.current_price}"],
                ['50 DMA', f"Rs {stock.dma_50}"],
                ['200 DMA', f"Rs {stock.dma_200}"],
                ['RSI', str(stock.rsi)],
                ['MACD Signal', stock.macd_signal],
                ['Support Level', f"Rs {stock.support_level}"],
                ['Resistance Level', f"Rs {stock.resistance_level}"],
                ['Volume Trend', stock.volume_trend],
            ]
            
            technical_table = Table(technical_data, colWidths=[3*inch, 3*inch])
            technical_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            elements.append(technical_table)
            elements.append(Spacer(1, 12))
            
            # Latest News
            elements.append(Paragraph("Latest News (Top 5)", heading_style))
            for news_idx, news in enumerate(stock.latest_news, 1):
                elements.append(Paragraph(f"{news_idx}. {news.title}", subheading_style))
                elements.append(Paragraph(f"Date: {news.date} | Source: {news.source}", styles['Normal']))
                elements.append(Paragraph(f"Summary: {news.summary}", styles['Normal']))
                elements.append(Paragraph(f"URL: {news.url}", styles['Normal']))
                elements.append(Spacer(1, 8))
            
            elements.append(Spacer(1, 12))
            
            # Trade Recommendation
            elements.append(Paragraph("Trade Recommendation", heading_style))
            rec = stock.trade_recommendation
            trade_data = [
                ['Parameter', 'Value'],
                ['Entry Zone', f"Rs {rec.entry_zone_min} - Rs {rec.entry_zone_max}"],
                ['Target 1', f"Rs {rec.target_1_price} ({rec.target_1_gain_percent}% gain) - {rec.target_1_timeframe}"],
                ['Target 2', f"Rs {rec.target_2_price} ({rec.target_2_gain_percent}% gain) - {rec.target_2_timeframe}"],
                ['Stop Loss', f"Rs {rec.stop_loss} ({rec.stop_loss_risk_percent}% risk)"],
                ['Risk-Reward Ratio', f"1:{rec.risk_reward_ratio}"],
                ['Position Size', f"{rec.position_size_percent}% of portfolio"],
                ['Time Horizon', f"{rec.time_horizon_weeks} weeks"],
            ]
            
            trade_table = Table(trade_data, colWidths=[3*inch, 3*inch])
            trade_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.lightgreen),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            elements.append(trade_table)
            elements.append(Spacer(1, 12))
            
            # Risks
            elements.append(Paragraph("Risks to Consider", heading_style))
            for risk in stock.risks:
                elements.append(Paragraph(f"• {risk}", styles['Normal']))
            elements.append(Spacer(1, 12))
            
            # Rationale
            elements.append(Paragraph("Why This Trade", heading_style))
            elements.append(Paragraph(stock.trade_rationale, styles['Normal']))
            
            # Page break after each stock (except the last one)
            if idx < len(analysis_data.top_stocks) - 1:
                elements.append(PageBreak())
        
        # Build PDF
        doc.build(elements)
        buffer.seek(0)
        return buffer
    
    except Exception as e:
        st.error(f"Error generating PDF: {str(e)}")
        return None

# Display stock analysis results
def display_stock_analysis(analysis_data, start_date, end_date):
    """
    Display the stock analysis results in a formatted way
    """
    st.markdown("---")
    st.header("🎯 AI-Powered Swing Trade Analysis")
    
    # Market Context
    with st.expander("📊 Market Context", expanded=False):
        market_ctx = analysis_data.market_context
        for key, value in market_ctx.items():
            st.markdown(f"**{key.replace('_', ' ').title()}:** {value}")
    
    st.markdown("---")
    
    # Store chart images for PDF
    charts_data = {}
    
    # Top Stocks Analysis
    for stock in analysis_data.top_stocks:
        with st.container():
            # Header with rank and score
            col1, col2 = st.columns([3, 1])
            with col1:
                st.subheader(f"🏆 RANK {stock.rank}: {stock.symbol} - {stock.company_name}")
            with col2:
                score_color = "🟢" if stock.overall_score >= 80 else "🟡" if stock.overall_score >= 60 else "🔴"
                st.metric("Overall Score", f"{stock.overall_score}/100", delta=score_color)
            
            # Fetch and display price chart
            st.markdown("### 📈 Price Chart & Technical Analysis")
            with st.spinner(f"Loading price data for {stock.symbol}..."):
                price_df, error = fetch_stock_price_data(stock.symbol, start_date, end_date)
                
                if price_df is not None and not price_df.empty:
                    fig = create_price_chart(price_df, stock.symbol)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Try to save chart for PDF with better error handling
                        try:
                            import plotly.io as pio
                            
                            # Set kaleido as the default renderer
                            pio.kaleido.scope.mathjax = None
                            
                            # Save the chart
                            chart_img = BytesIO()
                            fig.write_image(
                                chart_img, 
                                format='png', 
                                width=800, 
                                height=500,
                                engine='kaleido'
                            )
                            chart_img.seek(0)
                            charts_data[stock.symbol] = chart_img
                            
                        except ImportError as ie:
                            st.warning(f"⚠️ Kaleido not properly installed. Charts will not be included in PDF. Error: {str(ie)}")
                        except Exception as chart_error:
                            st.warning(f"⚠️ Could not save chart for PDF: {str(chart_error)}")
                            # Continue without chart in PDF
                            pass
                else:
                    st.warning(f"⚠️ Could not load price chart for {stock.symbol}: {error if error else 'No data available'}")
            
            # Create tabs for different sections
            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                "🏦 Institutional Activity", 
                "📊 Fundamentals",
                "📈 Price History",
                "📰 Latest News",
                "📉 Technical Setup", 
                "💡 Trade Plan"
            ])
            
            with tab1:
                st.markdown(f"**Institutional Activity:**\n{stock.institutional_activity}")
                st.metric("Total Institutional Buying", f"₹{stock.total_buying_cr} Cr")
                st.markdown("**Notable Buyers:**")
                for buyer in stock.notable_buyers:
                    st.markdown(f"- {buyer}")
            
            with tab2:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Market Cap", f"₹{stock.market_cap_cr} Cr")
                    st.metric("Sector", stock.sector)
                with col2:
                    st.metric("Revenue Growth (QoQ)", f"{stock.qoq_revenue_growth}%")
                    st.metric("Profit Growth (QoQ)", f"{stock.qoq_profit_growth}%")
                with col3:
                    st.metric("ROE", f"{stock.roe_percent}%")
                    st.metric("Debt/Equity", f"{stock.debt_equity_ratio}")
                
                st.info(f"**Key Catalyst:** {stock.key_catalyst}")
            
            with tab3:
                st.subheader("Price History Analysis")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Period High", f"₹{stock.price_history.period_high}")
                    st.metric("Period Low", f"₹{stock.price_history.period_low}")
                    st.metric("Start Price", f"₹{stock.price_history.period_start}")
                with col2:
                    st.metric("End Price", f"₹{stock.price_history.period_end}")
                    st.metric("Price Change", f"{stock.price_history.price_change_percent}%", 
                             delta=f"{stock.price_history.price_change_percent}%")
                    st.metric("Average Volume", f"{stock.price_history.average_volume:,.0f}")
            
            with tab4:
                st.subheader("Latest News (Top 5)")
                for idx, news in enumerate(stock.latest_news, 1):
                    with st.expander(f"{idx}. {news.title}"):
                        st.markdown(f"**Date:** {news.date}")
                        st.markdown(f"**Source:** {news.source}")
                        st.markdown(f"**Summary:** {news.summary}")
                        st.markdown(f"**URL:** [{news.url}]({news.url})")
            
            with tab5:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Current Price", f"₹{stock.current_price}")
                    st.metric("50 DMA", f"₹{stock.dma_50}")
                    st.metric("200 DMA", f"₹{stock.dma_200}")
                with col2:
                    st.metric("RSI", stock.rsi)
                    st.metric("MACD Signal", stock.macd_signal)
                with col3:
                    st.metric("Support", f"₹{stock.support_level}")
                    st.metric("Resistance", f"₹{stock.resistance_level}")
                
                st.markdown(f"**Volume Trend:** {stock.volume_trend}")
            
            with tab6:
                rec = stock.trade_recommendation
                
                # Entry and Targets
                st.markdown("### 📍 Entry & Targets")
                col1, col2 = st.columns(2)
                with col1:
                    st.success(f"**Entry Zone:** ₹{rec.entry_zone_min} - ₹{rec.entry_zone_max}")
                    st.info(f"**Target 1:** ₹{rec.target_1_price} ({rec.target_1_gain_percent}% gain) - {rec.target_1_timeframe}")
                    st.info(f"**Target 2:** ₹{rec.target_2_price} ({rec.target_2_gain_percent}% gain) - {rec.target_2_timeframe}")
                with col2:
                    st.error(f"**Stop Loss:** ₹{rec.stop_loss} ({rec.stop_loss_risk_percent}% risk)")
                    st.metric("Risk-Reward Ratio", f"1:{rec.risk_reward_ratio}")
                    st.metric("Position Size", f"{rec.position_size_percent}% of portfolio")
                    st.metric("Time Horizon", f"{rec.time_horizon_weeks} weeks")
                
                # Risks
                st.markdown("### ⚠️ Risks to Consider")
                for risk in stock.risks:
                    st.warning(f"- {risk}")
                
                # Rationale
                st.markdown("### 🎯 Why This Trade")
                st.markdown(stock.trade_rationale)
            
            st.markdown("---")
    
    # Excluded Stocks
    if analysis_data.excluded_stocks:
        with st.expander("❌ Excluded Stocks", expanded=False):
            for excluded in analysis_data.excluded_stocks:
                st.markdown(f"**{excluded.get('symbol', 'Unknown')}:** {excluded.get('reason', 'No reason provided')}")
    
    # PDF Export Button
    st.markdown("---")
    st.header("📄 Export Report")
    
    if st.button("📥 Generate PDF Report", type="primary", use_container_width=True):
        with st.spinner("Generating PDF report... This may take a moment..."):
            try:
                pdf_buffer = generate_pdf_report(analysis_data, start_date, end_date, charts_data)
                
                if pdf_buffer:
                    st.success("✅ PDF report generated successfully!")
                    
                    # Show info about charts
                    if charts_data:
                        st.info(f"📊 {len(charts_data)} price chart(s) included in the PDF")
                    else:
                        st.warning("⚠️ PDF generated without charts (kaleido issue)")
                    
                    st.download_button(
                        label="💾 Download PDF Report",
                        data=pdf_buffer,
                        file_name=f"SAST_Stock_Analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
            except Exception as e:
                st.error(f"Error generating PDF: {str(e)}")

# Main app
def main():
    st.title("📊 SAST Deal Tracker - Bulk Deals Monitor")
    st.markdown("### Track bulk deals for top 50 whale clients from NSE")
    
    # Load clients
    clients = load_clients()
    
    # Sidebar for controls
    st.sidebar.header("⚙️ Configuration")

    # API Key input - use secrets if available
    try:
        default_api_key = st.secrets.get("PERPLEXITY_API_KEY", "")
    except:
        default_api_key = ""

    api_key = st.sidebar.text_input(
        "Perplexity API Key", 
        value=default_api_key,
        type="password", 
        help="Enter your Perplexity API key for AI analysis"
    )
    
    st.sidebar.markdown("---")
    
    # Date range selection
    st.sidebar.header("📅 Date Range")
    date_option = st.sidebar.radio(
        "Select Date Range:",
        ["1 Day", "1 Week", "1 Month", "1 Year", "Custom"],
        index=1  # Default to 1 Week
    )
    
    # Custom date range
    if date_option == "Custom":
        col1, col2 = st.sidebar.columns(2)
        with col1:
            start_date_input = st.date_input("Start Date", datetime.now() - timedelta(days=7))
        with col2:
            end_date_input = st.date_input("End Date", datetime.now())
        
        start_date = start_date_input.strftime("%d-%m-%Y")
        end_date = end_date_input.strftime("%d-%m-%Y")
    else:
        start_date, end_date = get_date_range(date_option)
    
    st.sidebar.markdown(f"**From:** {start_date}")
    st.sidebar.markdown(f"**To:** {end_date}")
    
    st.sidebar.markdown("---")
    
    # Client filter
    st.sidebar.header("🏢 Client Filter")
    selected_clients = st.sidebar.multiselect(
        "Select Clients to Track:",
        options=["All"] + clients,
        default=["All"],
        help="Select specific clients or 'All' to view all whale clients"
    )
    
    st.sidebar.markdown("---")
    
    # Data source selection
    st.sidebar.header("📂 Data Source")
    data_source = st.sidebar.radio(
        "Choose data source:",
        ["Download from NSE", "Upload CSV File"],
        index=0
    )
    
    # File uploader for manual CSV upload
    uploaded_file = None
    if data_source == "Upload CSV File":
        uploaded_file = st.sidebar.file_uploader("Upload Bulk Deals CSV", type=['csv'])
    
    st.sidebar.markdown("---")
    
    # Fetch/Process data button
    if data_source == "Download from NSE":
        fetch_button = st.sidebar.button("🔍 Fetch Deals from NSE", type="primary", use_container_width=True)
    else:
        fetch_button = st.sidebar.button("📊 Process Uploaded CSV", type="primary", use_container_width=True, disabled=(uploaded_file is None))
    
    # Main content area
    if fetch_button:
        if data_source == "Upload CSV File" and uploaded_file is not None:
            # Process uploaded CSV file
            try:
                with st.spinner("Processing uploaded CSV file..."):
                    df = pd.read_csv(uploaded_file)
                error = None
            except Exception as e:
                df = None
                error = f"Error reading CSV file: {str(e)}"
        else:
            # Download from NSE
            with st.spinner("Downloading bulk deals CSV from NSE..."):
                df, error = fetch_bulk_deals(start_date, end_date)
        
        if error:
            st.error(error)
            st.info("💡 **Tip:** NSE API might be temporarily unavailable. Please try again in a few moments.")
        elif df is not None and not df.empty:
            # Save CSV file automatically
            save_folder = "saved_data"
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_filename = f"{save_folder}/bulk_deals_{start_date.replace('-', '')}_{end_date.replace('-', '')}_{timestamp}.csv"
            df.to_csv(csv_filename, index=False)
            
            # Debug: Show column names
            st.success(f"✅ Data loaded successfully! Found {len(df)} total records")
            st.success(f"💾 CSV file saved: `{csv_filename}`")
            with st.expander("🔍 Debug: View Column Names"):
                st.write("Columns in CSV:", df.columns.tolist())
                st.write("Sample data (first 3 rows):")
                st.dataframe(df.head(3))
            
            # Store data in session state to persist across interactions
            st.session_state['loaded_df'] = df
            st.session_state['start_date'] = start_date
            st.session_state['end_date'] = end_date
            
            # Filter by selected clients
            client_col = "Client Name "  # Note: has trailing space
            
            if "All" not in selected_clients and selected_clients:
                # Filter for specific selected clients
                mask = df[client_col].astype(str).str.upper().apply(
                    lambda x: any(client.upper() in x for client in selected_clients)
                )
                filtered_df = df[mask]
            else:
                # Show all data when "All" is selected OR filter for top 50 clients
                if "All" in selected_clients:
                    # Filter for any of the top 50 clients
                    mask = df[client_col].astype(str).str.upper().apply(
                        lambda x: any(client.upper() in x for client in clients)
                    )
                    filtered_df = df[mask]
                    
                    # If no matches found, show all data
                    if len(filtered_df) == 0:
                        st.warning(f"No matches found for top 50 clients. Showing all data.")
                        filtered_df = df
                else:
                    filtered_df = df
            
            st.session_state['filtered_df'] = filtered_df
        else:
            st.warning("No data available for the selected date range.")
    
    # Display data from session state if available (persists across button clicks)
    if 'filtered_df' in st.session_state and st.session_state['filtered_df'] is not None:
        df = st.session_state.get('loaded_df')
        filtered_df = st.session_state.get('filtered_df')
        start_date = st.session_state.get('start_date', start_date)
        end_date = st.session_state.get('end_date', end_date)
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Deals", len(df))
        with col2:
            st.metric("Filtered Deals", len(filtered_df))
        with col3:
            if len(filtered_df) > 0 and 'Quantity Traded' in filtered_df.columns:
                total_qty = filtered_df['Quantity Traded'].sum() if pd.api.types.is_numeric_dtype(filtered_df['Quantity Traded']) else 0
                st.metric("Total Quantity", f"{total_qty:,.0f}")
            else:
                st.metric("Total Quantity", "N/A")
        
        st.markdown("---")
        
        # Display data
        if len(filtered_df) > 0:
            st.subheader(f"📈 Bulk Deals ({len(filtered_df)} records)")
            
            # Download button
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f"bulk_deals_{start_date}_to_{end_date}.csv",
                mime="text/csv",
                use_container_width=False
            )
            
            st.markdown("---")
            
            # Pagination settings
            rows_per_page = 50
            total_rows = len(filtered_df)
            total_pages = (total_rows - 1) // rows_per_page + 1
            
            # Initialize session state for page number
            if 'current_page' not in st.session_state:
                st.session_state.current_page = 1
            
            # Pagination controls
            col1, col2, col3, col4, col5 = st.columns([1, 1, 2, 1, 1])
            
            with col1:
                if st.button("⏮️ First", use_container_width=True):
                    st.session_state.current_page = 1
                    st.rerun()
            
            with col2:
                if st.button("◀️ Previous", use_container_width=True):
                    if st.session_state.current_page > 1:
                        st.session_state.current_page -= 1
                        st.rerun()
            
            with col3:
                st.markdown(f"<h4 style='text-align: center;'>Page {st.session_state.current_page} of {total_pages}</h4>", unsafe_allow_html=True)
            
            with col4:
                if st.button("Next ▶️", use_container_width=True):
                    if st.session_state.current_page < total_pages:
                        st.session_state.current_page += 1
                        st.rerun()
            
            with col5:
                if st.button("Last ⏭️", use_container_width=True):
                    st.session_state.current_page = total_pages
                    st.rerun()
            
            # Calculate start and end indices for current page
            start_idx = (st.session_state.current_page - 1) * rows_per_page
            end_idx = min(start_idx + rows_per_page, total_rows)
            
            # Display current page info
            st.info(f"Showing records {start_idx + 1} to {end_idx} of {total_rows}")
            
            # Display paginated table
            page_df = filtered_df.iloc[start_idx:end_idx].reset_index(drop=True)
            
            st.dataframe(
                page_df,
                use_container_width=True,
                height=600
            )
            
            # AI Analysis Section
            st.markdown("---")
            st.header("🤖 AI-Powered Stock Analysis")
            
            col1, col2 = st.columns([2, 1])
            with col1:
                num_stocks = st.number_input(
                    "How many stocks do you want to analyze?",
                    min_value=1,
                    max_value=20,
                    value=5,
                    step=1,
                    help="Select the number of top stocks for AI analysis"
                )
            
            with col2:
                st.markdown("<br>", unsafe_allow_html=True)
                analyze_button = st.button(
                    "🚀 Analyze Stocks with AI",
                    type="primary",
                    use_container_width=True
                )
            
            additional_instructions = st.text_area(
                "Additional Instructions (Optional)",
                placeholder="E.g., Focus on small-cap stocks, avoid banking sector, look for stocks under ₹500, etc.",
                height=100,
                help="Provide any specific instructions or preferences for the AI analysis"
            )
            
            if analyze_button:
                if not api_key or api_key == "":
                    st.error("❌ Please provide a Perplexity API key in the sidebar!")
                else:
                    with st.spinner(f"🔍 Analyzing top {num_stocks} stocks using AI... This may take 30-60 seconds..."):
                        analysis_data, error = analyze_stocks_with_perplexity(
                            filtered_df,
                            num_stocks,
                            additional_instructions,
                            api_key,
                            start_date,
                            end_date
                        )
                    
                    if error:
                        st.error(f"❌ {error}")
                    elif analysis_data:
                        # Store analysis in session state
                        st.session_state['stock_analysis'] = analysis_data
                        st.success(f"✅ Analysis complete! Found {len(analysis_data.top_stocks)} swing trade opportunities.")
            
            # Display analysis if available
            if 'stock_analysis' in st.session_state:
                display_stock_analysis(st.session_state['stock_analysis'], start_date, end_date)
        else:
            st.warning("No deals found for the selected clients in this date range.")
    elif not fetch_button:
        # Instructions
        st.info("👈 **Configure your search parameters in the sidebar and click 'Fetch Deals' to start tracking!**")
        
        st.markdown("### 📋 Features:")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            - ✅ **Quick Date Ranges:** 1 Day, 1 Week, 1 Month, 1 Year
            - ✅ **Custom Date Range:** Select specific dates
            - ✅ **Client Filtering:** Track specific whale clients
            - ✅ **Download from NSE:** Automatic CSV download
            - ✅ **AI Stock Analysis:** Perplexity-powered swing trade recommendations
            - ✅ **Price Charts:** Interactive OHLC candlestick charts with moving averages
            """)
        
        with col2:
            st.markdown("""
            - ✅ **Upload CSV:** Manually upload bulk deals CSV
            - ✅ **CSV Export:** Export filtered data
            - ✅ **Top 50 Whales:** Pre-configured major clients
            - ✅ **Real-time Processing:** Instant data filtering
            - ✅ **Trade Recommendations:** Entry, targets, stop-loss levels
            - ✅ **PDF Export:** Generate comprehensive PDF reports with charts
            """)
        
        st.markdown("### 🏢 Top 50 Whale Clients Being Tracked:")
        
        # Display clients in columns
        num_cols = 3
        cols = st.columns(num_cols)
        for idx, client in enumerate(clients):
            with cols[idx % num_cols]:
                st.markdown(f"- {client}")

if __name__ == "__main__":
    main()


