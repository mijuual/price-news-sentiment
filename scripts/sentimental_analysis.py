import pandas as pd
import matplotlib.pyplot as plt
from textblob import TextBlob
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import seaborn as sns
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

class NewsAnalyzer:
    def __init__(self, news_df: pd.DataFrame):
        """
        Initialize with a financial news DataFrame.
        Expected columns: ['headline', 'url', 'publisher', 'date', 'stock']
        """
        self.df = news_df.copy()

    def overview(self):
        """
        Print basic dataset stats and clean null values from key columns.
        """
        print("🔍 Dataset Shape:", self.df.shape)
        print("\n First 5 Rows:\n", self.df.head())
        print("\n Last 5 Rows:\n", self.df.tail())
        print("\nℹ️ Dataset Info:")
        self.df.info()

        # Drop rows missing headline or date
        before = self.df.shape[0]
        self.df.dropna(subset=['headline', 'date'], inplace=True)
        after = self.df.shape[0]

        print(f"\n🧹 Removed {before - after} rows with missing 'headline' or 'date'.")
    
    def prepare_data(self):
        """
        Cleans and preprocesses the dataset:
        - Converts 'date' to datetime
        - Drops rows with missing headlines or dates
        - Ensures 'date' is the index for time series analysis
        """
        print("Initial shape:", self.df.shape)

        # Drop rows with missing values in critical columns
        self.df.dropna(subset=['headline', 'date'], inplace=True)

        # Convert 'date' to datetime with timezone stripping
        self.df['date'] = pd.to_datetime(self.df['date'], errors='coerce', format='mixed')
        self.df['date'] = self.df['date'].dt.tz_localize(None)

        # Set index
        self.df.set_index('date', inplace=True)

        print("After cleaning:", self.df.shape)
        print("Date range:", self.df.index.min(), "to", self.df.index.max())
    
    def parse_and_clean_dates(self, col='date', tickers=None):
        """
        Parse and clean the date column more flexibly to avoid over-filtering.

        Args:
            col (str): The name of the date column.
            tickers (list, optional): List of stock tickers to filter by.

        Returns:
            pd.DataFrame: Cleaned DataFrame with a datetime64[ns] column.
        """
        from dateutil import parser

        df = self.df.copy()
        if tickers:
            df = df[df['stock'].isin(tickers)]

        print(f"🔍 Raw row count: {len(df)}")

        # Step 1: Initial parsing
        parsed_main = pd.to_datetime(df[col], errors='coerce')

        # 🔁 Force tz-naive immediately
        parsed_main = parsed_main.dt.tz_localize(None)

        # Step 2: Fallback parse for unparsed rows
        still_na = parsed_main.isna()
        if still_na.sum() > 0:
            print(f"🔧 Trying to salvage {still_na.sum()} unparsed rows with row-by-row parsing...")

            def try_parse(x):
                try:
                    return parser.parse(x)
                except:
                    return pd.NaT

            fallback_parsed = df.loc[still_na, col].astype(str).apply(try_parse)
            fallback_parsed = pd.to_datetime(fallback_parsed, errors='coerce')
            fallback_parsed = fallback_parsed.dt.tz_localize(None)

            # ✅ Now assign safely
            parsed_main.loc[still_na] = fallback_parsed

        # Final cleanup and normalization
        invalid = parsed_main.isna().sum()
        df[col] = parsed_main
        df = df[df[col].notna()]
        df[col] = df[col].dt.normalize()
        df[col] = df[col].astype('datetime64[ns]')

        print(f"✅ Parsed {len(df)} rows (dropped {invalid} invalid).")
        print("📌 dtype of 'date' column after parsing:", df[col].dtype)

        return df




    
    def normalize_dates(self, df, col='date', inplace=False):
       
        """
        Normalize a datetime column (remove time, standardize dtype).

        Args:
            df (pd.DataFrame): DataFrame with a datetime column.
            col (str): Column name to normalize.
            inplace (bool): Whether to update self.df or return new.

        Returns:
            pd.DataFrame or None
        """
        df = df.copy()

        if col not in df.columns:
            raise KeyError(f"'{col}' column not found in DataFrame.")

        # Try to coerce to datetime (safe fallback)
        df[col] = pd.to_datetime(df[col], errors='coerce')

        if not pd.api.types.is_datetime64_any_dtype(df[col]):
            raise TypeError(f"The '{col}' column is not datetime-like.")

        df[col] = df[col].dt.normalize()
        df[col] = df[col].astype('datetime64[ns]')

        print(f"📅 Normalized '{col}' column. Sample:")
        print(df[[col]].head())

        if inplace:
            self.df = df
            return None
        else:
            return df




   

    def describe_content(self):
        """
        Prints descriptive statistics of the news dataset:
        - Headline length distribution
        - Top publishers
        - Unique stock tickers
        - Date range
        """
        # Display descriptive statistics for the 'headline' column
        print("Descriptive statistics for 'headline' column:")
        print(self.df['headline'].describe())
    

        # Top publishers
        print("\n🏢 Top 10 Publishers by Article Count:")
        print(self.df['publisher'].value_counts().head(10))

        # Unique stocks
        print(f"\n📈 Unique Stocks Mentioned: {self.df['stock'].nunique()}")
        print("Examples:", self.df['stock'].unique()[:10])

        # Date range
        print(f"\n🕒 Date Range: {self.df.index.min()} to {self.df.index.max()}")
    
    def extract_topics(self, n_topics=5, n_top_words=10):
        """
        Performs topic modeling on the headlines using LDA.

        Args:
            n_topics (int): Number of topics to extract.
            n_top_words (int): Number of top words to show per topic.
        """
        print(f"\n🧠 Extracting {n_topics} topics from headlines...")

        # Step 1: Drop NA and convert to list
        headlines = self.df['headline'].dropna().astype(str).tolist()

        # Step 2: Preprocess headlines
        stop_words = set(stopwords.words('english'))
        cleaned = []
        for line in headlines:
            line = line.lower()
            line = line.translate(str.maketrans('', '', string.punctuation))
            words = [w for w in line.split() if w not in stop_words and w.isalpha()]
            cleaned.append(' '.join(words))

        # Step 3: Vectorize
        vectorizer = CountVectorizer(max_df=0.9, min_df=5, stop_words='english')
        X = vectorizer.fit_transform(cleaned)

        # Step 4: Apply LDA
        lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
        lda.fit(X)

        # Step 5: Display Topics
        print("\n🗂️ Top Words per Topic:")
        feature_names = vectorizer.get_feature_names_out()
        for idx, topic in enumerate(lda.components_):
            top_words = [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
            print(f"\nTopic #{idx + 1}: {', '.join(top_words)}")

    
    def plot_publication_frequency(self, save_path=None):
        """
        Plots the number of articles published per day.
        Converts and cleans the 'date' column as needed.

        Args:
            save_path (str, optional): If provided, saves the plot to this path.
        """
        # Ensure 'date' column exists and is datetime
       # Clean and convert date
        self.df['date'] = pd.to_datetime(self.df['date'], errors='coerce', format='mixed')
        self.df = self.df[self.df['date'].notna()]
        self.df['date'] = self.df['date'].astype('datetime64[ns]')

        if pd.api.types.is_datetime64_any_dtype(self.df['date']):
            if self.df['date'].dt.tz is not None:
                self.df['date'] = self.df['date'].dt.tz_localize(None)
        else:
            raise TypeError("The 'date' column is not datetime-like.")

        # Set index and resample by day
        self.df.set_index('date', inplace=True)
        daily_counts = self.df.resample('D').size()

        # Plot
        plt.figure(figsize=(14, 6))
        plt.plot(daily_counts, color='blue', linewidth=1.5)
        plt.title("Article Publication Frequency Over Time", fontsize=16)
        plt.xlabel("Date")
        plt.ylabel("Number of Articles")
        plt.grid(True)
        plt.tight_layout()

        # Save or show
        if save_path:
            plt.savefig(save_path, dpi=300)
            print(f"✅ Plot saved to: {save_path}")
        plt.show()
    
    def analyze_sentiment(self, tickers=None, inplace=True):
        """
        Performs sentiment analysis on selected companies' headlines using TextBlob.

        Args:
            tickers (list or None): List of stock ticker symbols (e.g., ['AAPL', 'AMZN']).
                                    If None, all tickers are analyzed.
            inplace (bool): If True, updates self.df; otherwise returns filtered DataFrame.

        Returns:
            pd.DataFrame (optional): Sentiment-scored DataFrame if inplace=False
        """
        df = self.df.copy()

        # Filter by ticker symbol
        if tickers:
            tickers = [t.upper() for t in tickers]
            df = df[df['stock'].isin(tickers)]

        # Ensure 'headline' is string type
        df['headline'] = df['headline'].astype(str)

        # Apply sentiment analysis
        df['polarity'] = df['headline'].apply(lambda x: TextBlob(x).sentiment.polarity)
        df['subjectivity'] = df['headline'].apply(lambda x: TextBlob(x).sentiment.subjectivity)

        print(f"🧪 Sentiment analysis completed for {df['stock'].nunique()} stock(s), {len(df)} headlines.")

        # Return or update
        if inplace:
            self.df.update(df)
        else:
            return df
