import yfinance as yf
import pandas as pd
import ta
import matplotlib.pyplot as plt
import os
from docx import Document
from docx.shared import Inches
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

# Load NASDAQ-100 stock list from CSV
nasdaq100_df = pd.read_csv('nasdaq100.csv')
nasdaq100_stocks = nasdaq100_df['Symbol'].tolist()


# Function to calculate VFI
def calculate_vfi(close, high, low, volume):
    vfi = ((close - low) - (high - close)) / (high - low) * volume
    return vfi.cumsum()


# Function to fetch data and calculate indicators
def check_stocks(stocks):
    one_condition_stocks = {}
    two_conditions_stocks = {}
    three_conditions_stocks = {}
    four_conditions_stocks = {}
    five_conditions_stocks = {}

    for stock in stocks:
        # Fetch historical data for 1 hour interval
        data = yf.download(stock, period="3mo", interval="1h")

        if data.empty:
            continue

        # Resample data to 4-hour intervals
        data_4h = data.resample('4H').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()

        # Calculate Bollinger Bands with a window of 100 and 2 standard deviations on 4-hour data
        bb_indicator = ta.volatility.BollingerBands(close=data_4h['Close'], window=100, window_dev=2)
        data_4h['BB_High'] = bb_indicator.bollinger_hband()
        data_4h['BB_Low'] = bb_indicator.bollinger_lband()

        # Calculate RSI with a window of 10 on 4-hour data
        data_4h['RSI'] = ta.momentum.RSIIndicator(data_4h['Close'], window=10).rsi()

        # Calculate MACD with fast length 8, slow length 21, and signal smoothing 5 on 4-hour data
        macd_indicator = ta.trend.MACD(data_4h['Close'], window_slow=21, window_fast=8, window_sign=5)
        data_4h['MACD'] = macd_indicator.macd()
        data_4h['MACD_Signal'] = macd_indicator.macd_signal()

        # Calculate VFI
        data_4h['VFI'] = calculate_vfi(data_4h['Close'], data_4h['High'], data_4h['Low'], data_4h['Volume'])

        # Calculate VPT
        data_4h['VPT'] = ta.volume.VolumePriceTrendIndicator(data_4h['Close'], data_4h['Volume']).volume_price_trend()

        # Check conditions
        latest_rsi = data_4h['RSI'].iloc[-1]
        latest_close = data_4h['Close'].iloc[-1]
        bb_low = data_4h['BB_Low'].iloc[-1]
        latest_macd = data_4h['MACD'].iloc[-1]
        latest_macd_signal = data_4h['MACD_Signal'].iloc[-1]
        latest_vfi = data_4h['VFI'].iloc[-1]
        latest_vpt = data_4h['VPT'].iloc[-1]

        # Condition checks
        rsi_condition = latest_rsi < 30
        bb_condition = latest_close < bb_low
        macd_condition = latest_macd > latest_macd_signal and data_4h['MACD'].iloc[-2] <= data_4h['MACD_Signal'].iloc[
            -2]
        vfi_condition = latest_vfi > 0
        vpt_condition = latest_vpt > data_4h['VPT'].iloc[-2]  # Indicates an increase in VPT, suggesting more buying

        # Categorize stocks based on conditions met and store conditions
        conditions_met = sum([rsi_condition, bb_condition, macd_condition, vfi_condition, vpt_condition])
        conditions = []
        if rsi_condition:
            conditions.append("RSI below 30")
        if bb_condition:
            conditions.append("Price below lower Bollinger Band")
        if macd_condition:
            conditions.append("MACD buy signal")
        if vfi_condition:
            conditions.append("VFI breaks above 0")
        if vpt_condition:
            conditions.append("VPT increasing (buying pressure)")

        if conditions_met == 1:
            one_condition_stocks[stock] = (data_4h, conditions)
        elif conditions_met == 2:
            two_conditions_stocks[stock] = (data_4h, conditions)
        elif conditions_met == 3:
            three_conditions_stocks[stock] = (data_4h, conditions)
        elif conditions_met == 4:
            four_conditions_stocks[stock] = (data_4h, conditions)
        elif conditions_met == 5:
            five_conditions_stocks[stock] = (data_4h, conditions)

    return one_condition_stocks, two_conditions_stocks, three_conditions_stocks, four_conditions_stocks, five_conditions_stocks


# Function to plot the chart only when at least two conditions are met
def plot_chart_if_two_conditions_met(data, conditions, stock_symbol, document):
    if len(conditions) >= 2:
        fig, ax = plt.subplots(5, 1, figsize=(12, 12))
        fig.suptitle(f"Stock: {stock_symbol} | Conditions: {', '.join(conditions)}", fontsize=16)

        # Plot Close price and Bollinger Bands
        ax[0].plot(data.index, data['Close'], label='Close Price', color='blue')
        ax[0].plot(data.index, data['BB_High'], label='Upper Bollinger Band', color='red', linestyle='--')
        ax[0].plot(data.index, data['BB_Low'], label='Lower Bollinger Band', color='green', linestyle='--')
        ax[0].set_ylabel('Price')
        ax[0].legend()

        # Plot RSI
        ax[1].plot(data.index, data['RSI'], label='RSI', color='orange')
        ax[1].axhline(y=30, color='r', linestyle='--')
        ax[1].set_ylabel('RSI')
        ax[1].legend()

        # Plot MACD and Signal line
        ax[2].plot(data.index, data['MACD'], label='MACD', color='purple')
        ax[2].plot(data.index, data['MACD_Signal'], label='Signal line', color='cyan')
        ax[2].axhline(y=0, color='gray', linestyle='--')
        ax[2].set_ylabel('MACD')
        ax[2].legend()

        # Plot VFI
        ax[3].plot(data.index, data['VFI'], label='VFI', color='brown')
        ax[3].axhline(y=0, color='gray', linestyle='--')
        ax[3].set_ylabel('VFI')
        ax[3].legend()

        # Plot VPT
        ax[4].plot(data.index, data['VPT'], label='VPT', color='magenta')
        ax[4].set_ylabel('VPT')
        ax[4].legend()

        plt.tight_layout()

        # Save plot to a file
        chart_filename = f"{stock_symbol}_chart.png"
        plt.savefig(chart_filename)
        plt.close(fig)

        # Add chart to the document
        document.add_heading(f"Stock: {stock_symbol}", level=2)
        document.add_paragraph(f"Conditions: {', '.join(conditions)}")
        document.add_picture(chart_filename, width=Inches(6))

        # Remove the saved chart file
        os.remove(chart_filename)


# Function to run the check once and save results to a Word document
def run_check_once(stocks):
    # Create a new Word document
    document = Document()
    document.add_heading('Stock Analysis Report', 0)

    # Fetch data and check conditions for each stock
    one_condition_stocks, two_conditions_stocks, three_conditions_stocks, four_conditions_stocks, five_conditions_stocks = check_stocks(
        stocks)

    # Display charts only for stocks meeting at least two conditions
    for stock, (data, conditions) in two_conditions_stocks.items():
        plot_chart_if_two_conditions_met(data, conditions, stock, document)

    for stock, (data, conditions) in three_conditions_stocks.items():
        plot_chart_if_two_conditions_met(data, conditions, stock, document)

    for stock, (data, conditions) in four_conditions_stocks.items():
        plot_chart_if_two_conditions_met(data, conditions, stock, document)

    for stock, (data, conditions) in five_conditions_stocks.items():
        plot_chart_if_two_conditions_met(data, conditions, stock, document)

    # Save the document
    document.save('Stock_Analysis_Report.docx')
    print("Stock analysis report saved to 'Stock_Analysis_Report.docx'.")


# Main function
if __name__ == "__main__":
    custom_stocks_input = input("Enter a comma-separated list of stock symbols (or press Enter to use NASDAQ-100): ")
    if custom_stocks_input:
        custom_stocks = [s.strip() for s in custom_stocks_input.split(',')]
        run_check_once(custom_stocks)
    else:
        run_check_once(nasdaq100_stocks)
