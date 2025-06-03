# Financial News Sentiment Analysis and Stock Movement Correlation
This project explores the relationship between financial news sentiment and stock price movements. By leveraging natural language processing (NLP), technical indicators, and financial datasets, it aims to enhance predictive analytics and support investment decision-making at Nova Financial Solutions.

## Project Overview
Nova Financial Solutions seeks to improve financial forecasting by integrating insights from financial news. This project:

Analyzes news headlines to extract sentiment using NLP techniques.

Applies technical analysis indicators (SMA, RSI, MACD) on stock price data using TA-Lib.

Visualizes trends and indicators for exploratory data analysis.

Investigates the correlation between sentiment and stock price returns.

## Methodology

### Task 1: News Dataset Analysis
Cleaned and prepared news data.

Performed descriptive statistics (headline length, publisher frequency).

Conducted topic modeling to identify frequent financial terms.

Analyzed publication frequency over time.

### Task 2: Stock Price Analysis
Used Apple (AAPL) historical price data.

Applied technical indicators using TA-Lib: SMA, RSI, MACD.

Visualized indicators and price movements.

### Task 3: Sentiment–Price Correlation 
Sentiment scores for financial news headlines were aligned with stock price data.

Correlation Analysis: Statistical methods, such as Pearson, was used to analyze the relationship between sentiment and stock price returns.

Result: A moderate correlation was found between positive sentiment and stock price increases, with a stronger effect observed for certain technical indicators (e.g., RSI and SMA).

This indicates that financial news sentiment can provide predictive insights when combined with technical indicators for stock movement forecasting.

## Technologies Used
* Python 3.10+

* Pandas / NumPy – data manipulation

* TA-Lib – technical indicators

* NLTK / TextBlob – sentiment analysis

* Plotly / Matplotlib / Seaborn – data visualization

* Git / GitHub – version control

## Conclusion
This project provides a robust analytical framework for Nova Financial Solutions to harness the predictive potential of financial news sentiment. By integrating sentiment analysis with technical stock indicators, the study reveals valuable correlations that can inform more strategic and data-driven investment decisions. With all key tasks completed, Nova is now equipped to further refine its forecasting models and enhance its decision-making process in dynamic financial markets.