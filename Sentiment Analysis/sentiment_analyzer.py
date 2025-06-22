
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
import re
import string
from collections import Counter
from wordcloud import WordCloud
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class SentimentAnalyzer:
    def __init__(self, data_path=None, dataframe=None, text_column='text'):
        """
        Initialize the Sentiment Analyzer

        Args:
            data_path (str): Path to CSV file
            dataframe (pd.DataFrame): DataFrame with text data
            text_column (str): Name of the column containing text
        """
        if data_path:
            self.df = pd.read_csv(data_path)
        elif dataframe is not None:
            self.df = dataframe.copy()
        else:
            # Create sample dataset for demonstration
            self.df = self.create_sample_data()

        self.text_column = text_column
        self.sia = SentimentIntensityAnalyzer()
        self.stop_words = set(stopwords.words('english'))

    def create_sample_data(self):
        """
        Create sample text data for sentiment analysis demo
        """
        sample_reviews = [
            "This product is absolutely amazing! I love it so much and would definitely recommend it to everyone.",
            "Terrible quality and poor customer service. Very disappointed with my purchase.",
            "It's okay, nothing special but does the job. Average product overall.",
            "Outstanding quality and fast delivery. Exceeded my expectations completely!",
            "Waste of money. The product broke after just one day of use.",
            "Great value for money. Really satisfied with this purchase and the quality.",
            "Not what I expected. The description was misleading and the product is cheaply made.",
            "Excellent customer service and high-quality product. Will buy again!",
            "Mediocre product. It works but there are better alternatives available.",
            "Fantastic! This is exactly what I was looking for. Perfect quality and design.",
            "Poor packaging and the product arrived damaged. Very unsatisfied.",
            "Good product overall. Some minor issues but mostly satisfied with the purchase.",
            "Amazing quality and great price. Highly recommend to anyone looking for this type of product.",
            "Completely useless. Don't waste your time or money on this junk.",
            "Decent product but overpriced. You can find better deals elsewhere.",
            "Love this product! Works perfectly and arrived quickly. Five stars!",
            "Disappointing. The product doesn't work as advertised and customer support is unhelpful.",
            "Pretty good quality and reasonable price. Would consider buying again.",
            "Exceptional product with outstanding features. Best purchase I've made this year!",
            "Awful experience. Product is defective and return process is complicated.",
            "It's fine. Does what it's supposed to do. Nothing extraordinary.",
            "Brilliant product! Excellent quality and very user-friendly. Highly satisfied.",
            "Total disaster. Product is completely different from what was shown in pictures.",
            "Good enough for the price. Not the best quality but acceptable.",
            "Perfect! Exactly as described and works flawlessly. Very happy with this purchase.",
            "Horrible quality and terrible design. Regret buying this product.",
            "Solid product with good features. Meets my expectations and works well.",
            "Extraordinary quality and amazing customer service. Will definitely shop here again!",
            "Unacceptable quality. Product is poorly made and doesn't last long.",
            "Nice product with some useful features. Overall a decent purchase."
        ]

        # Add some metadata
        products = ['Smartphone', 'Laptop', 'Headphones', 'Camera', 'Tablet']
        ratings = []

        data = []
        for i, review in enumerate(sample_reviews):
            # Generate rating based on sentiment for realism
            if any(word in review.lower() for word in ['amazing', 'outstanding', 'excellent', 'fantastic', 'brilliant', 'perfect', 'extraordinary']):
                rating = np.random.choice([4, 5], p=[0.2, 0.8])
            elif any(word in review.lower() for word in ['terrible', 'awful', 'horrible', 'disaster', 'useless', 'waste']):
                rating = np.random.choice([1, 2], p=[0.7, 0.3])
            else:
                rating = np.random.choice([2, 3, 4], p=[0.2, 0.6, 0.2])

            data.append({
                'text': review,
                'product': np.random.choice(products),
                'rating': rating,
                'review_id': f'REV_{i+1:03d}',
                'date': pd.date_range('2024-01-01', periods=30, freq='D')[i % 30]
            })

        return pd.DataFrame(data)

    def clean_text(self, text):
        """
        Clean and preprocess text data

        Args:
            text (str): Raw text to clean

        Returns:
            str: Cleaned text
        """
        if pd.isna(text):
            return ""

        # Convert to lowercase
        text = text.lower()

        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)

        # Remove extra whitespace
        text = ' '.join(text.split())

        return text

    def get_textblob_sentiment(self, text):
        """
        Get sentiment using TextBlob

        Args:
            text (str): Text to analyze

        Returns:
            dict: Sentiment scores and label
        """
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity

        # Determine sentiment label
        if polarity > 0.1:
            sentiment = 'Positive'
        elif polarity < -0.1:
            sentiment = 'Negative'
        else:
            sentiment = 'Neutral'

        return {
            'polarity': polarity,
            'subjectivity': subjectivity,
            'sentiment': sentiment
        }

    def get_vader_sentiment(self, text):
        """
        Get sentiment using VADER

        Args:
            text (str): Text to analyze

        Returns:
            dict: VADER sentiment scores and label
        """
        scores = self.sia.polarity_scores(text)

        # Determine sentiment label based on compound score
        if scores['compound'] >= 0.05:
            sentiment = 'Positive'
        elif scores['compound'] <= -0.05:
            sentiment = 'Negative'
        else:
            sentiment = 'Neutral'

        return {
            'compound': scores['compound'],
            'positive': scores['pos'],
            'negative': scores['neg'],
            'neutral': scores['neu'],
            'sentiment': sentiment
        }

    def analyze_sentiment(self):
        """
        Perform sentiment analysis on the dataset
        """
        print("Performing sentiment analysis...")

        # Clean text
        self.df['cleaned_text'] = self.df[self.text_column].apply(self.clean_text)

        # TextBlob sentiment analysis
        textblob_results = self.df['cleaned_text'].apply(self.get_textblob_sentiment)
        self.df['tb_polarity'] = textblob_results.apply(lambda x: x['polarity'])
        self.df['tb_subjectivity'] = textblob_results.apply(lambda x: x['subjectivity'])
        self.df['tb_sentiment'] = textblob_results.apply(lambda x: x['sentiment'])

        # VADER sentiment analysis
        vader_results = self.df[self.text_column].apply(self.get_vader_sentiment)
        self.df['vader_compound'] = vader_results.apply(lambda x: x['compound'])
        self.df['vader_positive'] = vader_results.apply(lambda x: x['positive'])
        self.df['vader_negative'] = vader_results.apply(lambda x: x['negative'])
        self.df['vader_neutral'] = vader_results.apply(lambda x: x['neutral'])
        self.df['vader_sentiment'] = vader_results.apply(lambda x: x['sentiment'])

        print("Sentiment analysis complete!")
        return self.df

    def create_sentiment_visualizations(self):
        """
        Create visualizations for sentiment analysis results
        """
        print("Creating sentiment visualizations...")

        # Set up the plotting area
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Sentiment Analysis Results', fontsize=16, fontweight='bold')

        # 1. Sentiment Distribution (TextBlob)
        tb_counts = self.df['tb_sentiment'].value_counts()
        colors = ['green', 'red', 'gray']
        axes[0, 0].pie(tb_counts.values, labels=tb_counts.index, autopct='%1.1f%%',
                      colors=colors, startangle=90)
        axes[0, 0].set_title('TextBlob Sentiment Distribution', fontweight='bold')

        # 2. Sentiment Distribution (VADER)
        vader_counts = self.df['vader_sentiment'].value_counts()
        axes[0, 1].pie(vader_counts.values, labels=vader_counts.index, autopct='%1.1f%%',
                      colors=colors, startangle=90)
        axes[0, 1].set_title('VADER Sentiment Distribution', fontweight='bold')

        # 3. Polarity Distribution
        axes[0, 2].hist(self.df['tb_polarity'], bins=20, alpha=0.7, color='blue', edgecolor='black')
        axes[0, 2].axvline(self.df['tb_polarity'].mean(), color='red', linestyle='--', 
                          label=f'Mean: {self.df["tb_polarity"].mean():.3f}')
        axes[0, 2].set_title('TextBlob Polarity Distribution', fontweight='bold')
        axes[0, 2].set_xlabel('Polarity Score')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].legend()

        # 4. VADER Compound Score Distribution
        axes[1, 0].hist(self.df['vader_compound'], bins=20, alpha=0.7, color='purple', edgecolor='black')
        axes[1, 0].axvline(self.df['vader_compound'].mean(), color='red', linestyle='--',
                          label=f'Mean: {self.df["vader_compound"].mean():.3f}')
        axes[1, 0].set_title('VADER Compound Score Distribution', fontweight='bold')
        axes[1, 0].set_xlabel('Compound Score')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].legend()

        # 5. Sentiment by Product
        if 'product' in self.df.columns:
            product_sentiment = pd.crosstab(self.df['product'], self.df['vader_sentiment'])
            product_sentiment.plot(kind='bar', ax=axes[1, 1], color=['red', 'gray', 'green'])
            axes[1, 1].set_title('Sentiment Distribution by Product', fontweight='bold')
            axes[1, 1].set_xlabel('Product')
            axes[1, 1].set_ylabel('Count')
            axes[1, 1].legend(title='Sentiment')
            axes[1, 1].tick_params(axis='x', rotation=45)

        # 6. Polarity vs Subjectivity
        sentiment_colors = {'Positive': 'green', 'Negative': 'red', 'Neutral': 'gray'}
        for sentiment in self.df['tb_sentiment'].unique():
            sentiment_data = self.df[self.df['tb_sentiment'] == sentiment]
            axes[1, 2].scatter(sentiment_data['tb_polarity'], sentiment_data['tb_subjectivity'],
                             label=sentiment, alpha=0.6, color=sentiment_colors[sentiment])
        axes[1, 2].set_title('Polarity vs Subjectivity', fontweight='bold')
        axes[1, 2].set_xlabel('Polarity')
        axes[1, 2].set_ylabel('Subjectivity')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('sentiment_analysis_plots.png', dpi=300, bbox_inches='tight')
        plt.show()

    def create_word_clouds(self):
        """
        Create word clouds for different sentiment categories
        """
        print("Creating word clouds...")

        # Separate text by sentiment
        positive_text = ' '.join(self.df[self.df['vader_sentiment'] == 'Positive'][self.text_column])
        negative_text = ' '.join(self.df[self.df['vader_sentiment'] == 'Negative'][self.text_column])
        neutral_text = ' '.join(self.df[self.df['vader_sentiment'] == 'Neutral'][self.text_column])

        # Create word clouds
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Word Clouds by Sentiment', fontsize=16, fontweight='bold')

        # Positive word cloud
        if positive_text:
            wordcloud_pos = WordCloud(width=400, height=300, background_color='white',
                                     colormap='Greens', stopwords=self.stop_words).generate(positive_text)
            axes[0].imshow(wordcloud_pos, interpolation='bilinear')
            axes[0].set_title('Positive Sentiment', fontweight='bold', color='green')
            axes[0].axis('off')

        # Negative word cloud
        if negative_text:
            wordcloud_neg = WordCloud(width=400, height=300, background_color='white',
                                     colormap='Reds', stopwords=self.stop_words).generate(negative_text)
            axes[1].imshow(wordcloud_neg, interpolation='bilinear')
            axes[1].set_title('Negative Sentiment', fontweight='bold', color='red')
            axes[1].axis('off')

        # Neutral word cloud
        if neutral_text:
            wordcloud_neu = WordCloud(width=400, height=300, background_color='white',
                                     colormap='Greys', stopwords=self.stop_words).generate(neutral_text)
            axes[2].imshow(wordcloud_neu, interpolation='bilinear')
            axes[2].set_title('Neutral Sentiment', fontweight='bold', color='gray')
            axes[2].axis('off')

        plt.tight_layout()
        plt.savefig('word_clouds.png', dpi=300, bbox_inches='tight')
        plt.show()

    def sentiment_summary_report(self):
        """
        Generate a comprehensive sentiment analysis report
        """
        print("\n" + "=" * 60)
        print("SENTIMENT ANALYSIS SUMMARY REPORT")
        print("=" * 60)

        # Basic statistics
        total_reviews = len(self.df)
        print(f"Total Reviews Analyzed: {total_reviews}")
        print()

        # TextBlob results
        print("TextBlob Analysis Results:")
        tb_sentiment_dist = self.df['tb_sentiment'].value_counts(normalize=True) * 100
        for sentiment, percentage in tb_sentiment_dist.items():
            print(f"  {sentiment}: {percentage:.1f}%")
        print(f"  Average Polarity: {self.df['tb_polarity'].mean():.3f}")
        print(f"  Average Subjectivity: {self.df['tb_subjectivity'].mean():.3f}")
        print()

        # VADER results
        print("VADER Analysis Results:")
        vader_sentiment_dist = self.df['vader_sentiment'].value_counts(normalize=True) * 100
        for sentiment, percentage in vader_sentiment_dist.items():
            print(f"  {sentiment}: {percentage:.1f}%")
        print(f"  Average Compound Score: {self.df['vader_compound'].mean():.3f}")
        print()

        # Most positive and negative reviews
        most_positive = self.df.loc[self.df['vader_compound'].idxmax()]
        most_negative = self.df.loc[self.df['vader_compound'].idxmin()]

        print("Most Positive Review:")
        print(f"  Text: {most_positive[self.text_column][:100]}...")
        print(f"  VADER Score: {most_positive['vader_compound']:.3f}")
        print()

        print("Most Negative Review:")
        print(f"  Text: {most_negative[self.text_column][:100]}...")
        print(f"  VADER Score: {most_negative['vader_compound']:.3f}")
        print()

        # Save results to CSV
        self.df.to_csv('sentiment_analysis_results.csv', index=False)
        print("Results saved to sentiment_analysis_results.csv")

        return {
            'total_reviews': total_reviews,
            'textblob_distribution': tb_sentiment_dist.to_dict(),
            'vader_distribution': vader_sentiment_dist.to_dict(),
            'avg_polarity': self.df['tb_polarity'].mean(),
            'avg_subjectivity': self.df['tb_subjectivity'].mean(),
            'avg_compound': self.df['vader_compound'].mean()
        }

    def run_complete_analysis(self):
        """
        Run the complete sentiment analysis pipeline
        """
        print("=" * 60)
        print("STARTING COMPLETE SENTIMENT ANALYSIS")
        print("=" * 60)

        # Perform sentiment analysis
        self.analyze_sentiment()

        # Create visualizations
        self.create_sentiment_visualizations()
        self.create_word_clouds()

        # Generate summary report
        summary = self.sentiment_summary_report()

        print("\n" + "=" * 60)
        print("SENTIMENT ANALYSIS COMPLETE!")
        print("Files generated:")
        print("- sentiment_analysis_plots.png")
        print("- word_clouds.png")
        print("- sentiment_analysis_results.csv")
        print("=" * 60)

        return summary

def main():
    """
    Main function to run sentiment analysis
    """
    print("Starting Sentiment Analysis...")

    # Initialize analyzer with sample data
    analyzer = SentimentAnalyzer()

    # Run complete analysis
    analyzer.run_complete_analysis()

if __name__ == "__main__":
    main()
