import pandas as pd
import talib
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import sys
print("Running with Python from:", sys.executable)

class StockAnalyzer:
    def __init__(self, price_df: pd.DataFrame):
        """
        Initialize with a stock price DataFrame.
        The DataFrame must include a 'Date' column.
        """
        self.df = price_df.copy()
    
    def prepare_data(self):
        """
        Performs initial data checks and preparation:
        - Prints the first few rows
        - Checks for missing values
        - Converts 'Date' column to datetime
        - Sets 'Date' as the index
        """
        print("First few rows of the dataset:")
        print(self.df.head())

        print("\nMissing values in each column:")
        print(self.df.isnull().sum())

        # Convert and set index
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        self.df.set_index('Date', inplace=True)
        print("\nDatetime conversion and index set completed.")

    def add_technical_indicators(self):
        """
        Calculates standard technical indicators using TA-Lib:
        - SMA (20-day and 50-day)
        - RSI (14-day)
        - MACD (default params)
        """
        self.df['SMA_20'] = talib.SMA(self.df['Close'], timeperiod=20)
        self.df['SMA_50'] = talib.SMA(self.df['Close'], timeperiod=50)
        self.df['RSI_14'] = talib.RSI(self.df['Close'], timeperiod=14)
        self.df['MACD'], self.df['MACD_signal'], self.df['MACD_hist'] = talib.MACD(self.df['Close'])

        print("\nTechnical indicators added. Sample:")
        print(self.df[['Close', 'SMA_20', 'SMA_50', 'RSI_14', 'MACD', 'MACD_signal']].tail())

    def plot_indicators(self, save_path=None):
        """
        Plots:
        - Close price with SMA indicators
        - RSI plot with 70/30 threshold lines
        Optionally saves the RSI plot if save_path is provided.
        """
        # Ensure index is datetime
        self.df.index = pd.to_datetime(self.df.index)

        # --- PRICE + SMA ---
        plt.figure(figsize=(16, 8))
        plt.plot(self.df['Close'], label='Close Price', color='black')
        plt.plot(self.df['SMA_20'], label='SMA 20', color='blue')
        plt.plot(self.df['SMA_50'], label='SMA 50', color='orange')
        plt.title('Stock Price with SMA Indicators')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # --- RSI Plot ---
        plt.figure(figsize=(16, 6))
        plt.plot(self.df['RSI_14'], label='RSI (14)', color='purple')
        plt.axhline(70, linestyle='--', color='red', alpha=0.5)
        plt.axhline(30, linestyle='--', color='green', alpha=0.5)
        plt.title('Relative Strength Index (RSI)')
        plt.xlabel('Date')
        plt.ylabel('RSI')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300)
            print(f"RSI plot saved to: {save_path}")

        plt.show()

    def plot_candlestick(self, save_path=None):
        """
        Creates a candlestick chart with SMA 50 overlay using Plotly.
        Optionally saves the plot as a PNG image.
        
        Args:
            save_path (str): Optional file path to save the chart image.
        """
        fig = go.Figure()

        # Add candlestick
        fig.add_trace(go.Candlestick(
            x=self.df.index,
            open=self.df['Open'],
            high=self.df['High'],
            low=self.df['Low'],
            close=self.df['Close'],
            name='OHLC'
        ))

        # Add SMA 50
        fig.add_trace(go.Scatter(
            x=self.df.index,
            y=self.df['SMA_50'],
            mode='lines',
            name='SMA 50',
            line=dict(color='red')
        ))

        # Layout
        fig.update_layout(
            title='Stock Price with SMA 50',
            yaxis_title='Price',
            xaxis_title='Date',
            template='plotly_white',
            height=500,
            xaxis_rangeslider_visible=False
        )

        # Save if path is given
        if save_path:
            fig.write_image(save_path, width=1000, height=500, scale=2)
            print(f"Candlestick chart saved to: {save_path}")

        fig.show()
    
    def plot_volume(self, save_path=None):
        """
        Plots a bar chart of trading volume over time.
        Optionally saves the chart as a PNG file.

        Args:
            save_path (str): File path to save the plot image (optional).
        """
        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=self.df.index,
            y=self.df['Volume'],
            name='Volume',
            marker_color='lightgray'
        ))

        fig.update_layout(
            title='Trading Volume',
            yaxis_title='Volume',
            xaxis_title='Date',
            template='plotly_white',
            height=300
        )

        if save_path:
            fig.write_image(save_path, width=1000, height=300, scale=2)
            print(f"Volume chart saved to: {save_path}")

        fig.show()


