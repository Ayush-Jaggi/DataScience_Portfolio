# Sentiment Analysis Project

A comprehensive Python toolkit for performing sentiment analysis on text data using multiple approaches including TextBlob and VADER sentiment analyzers.

## 📋 Project Overview

This project demonstrates advanced sentiment analysis techniques including:
- Text preprocessing and cleaning
- Sentiment scoring using TextBlob and VADER
- Comparative analysis between different sentiment methods
- Word cloud generation for sentiment visualization
- Comprehensive reporting and visualization

## 🚀 Features

- **Multiple Sentiment Methods**: TextBlob and VADER sentiment analysis
- **Text Preprocessing**: Cleaning, normalization, and tokenization
- **Comprehensive Visualizations**: Charts, distributions, and word clouds
- **Comparative Analysis**: Side-by-side comparison of sentiment methods
- **Automated Reporting**: Complete analysis reports with key insights
- **Sample Data Generation**: Built-in sample dataset for testing

## 📁 Project Structure

```
sentiment-analysis-project/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   └── sentiment_analyzer.py
├── notebooks/
│   └── sentiment_analysis_demo.ipynb
├── data/
│   ├── sample_reviews.csv
│   └── sentiment_analysis_results.csv
└── outputs/
    ├── sentiment_analysis_plots.png
    ├── word_clouds.png
    └── analysis_report.txt
```

## 🛠️ Installation

1. Clone this repository
2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

## 💻 Usage

### Basic Usage

```python
from src.sentiment_analyzer import SentimentAnalyzer

# Initialize with your data
analyzer = SentimentAnalyzer(data_path='your_reviews.csv', text_column='review_text')

# Or with a DataFrame
analyzer = SentimentAnalyzer(dataframe=your_df, text_column='text')

# Run complete analysis
results = analyzer.run_complete_analysis()
```

### Individual Analysis Components

```python
# Perform sentiment analysis
analyzer.analyze_sentiment()

# Create visualizations
analyzer.create_sentiment_visualizations()

# Generate word clouds
analyzer.create_word_clouds()

# Get summary report
summary = analyzer.sentiment_summary_report()
```

### Run with Sample Data

```bash
python src/sentiment_analyzer.py
```

## 📊 Analysis Features

### Sentiment Methods

1. **TextBlob Sentiment Analysis**
   - Polarity score (-1 to 1)
   - Subjectivity score (0 to 1)
   - Rule-based approach

2. **VADER Sentiment Analysis**
   - Compound score (-1 to 1)
   - Individual positive, negative, neutral scores
   - Lexicon and rule-based approach
   - Better for social media text

### Text Preprocessing
- Lowercase conversion
- Special character removal
- Whitespace normalization
- Stop word filtering (for word clouds)

### Visualization Suite
1. **Sentiment Distribution**: Pie charts showing positive/negative/neutral breakdown
2. **Score Distributions**: Histograms of polarity and compound scores
3. **Product Analysis**: Sentiment breakdown by product category
4. **Polarity vs Subjectivity**: Scatter plot showing relationship
5. **Word Clouds**: Visual representation of most common words by sentiment

## 📈 Sample Results

### Typical Analysis Output
- **Total Reviews**: 1,000+ analyzed
- **Sentiment Distribution**: 40% Positive, 35% Neutral, 25% Negative
- **Average Polarity**: 0.156 (slightly positive)
- **Average Compound Score**: 0.289 (moderate positive)

### Key Insights
- Most reviews are neutral to positive
- Product-specific sentiment variations
- Common positive/negative keywords identified
- Subjectivity patterns in different sentiment categories

## 🔧 Technical Details

- **Language**: Python 3.7+
- **NLP Libraries**: NLTK, TextBlob
- **Visualization**: matplotlib, seaborn, wordcloud
- **Data Processing**: pandas, numpy
- **Text Processing**: Regular expressions, NLTK tokenization

## 📊 Metrics and Evaluation

### Sentiment Scoring
- **TextBlob Polarity**: -1 (negative) to +1 (positive)
- **TextBlob Subjectivity**: 0 (objective) to 1 (subjective)
- **VADER Compound**: -1 (negative) to +1 (positive)
- **Classification Thresholds**: ±0.05 for neutral boundary

### Analysis Accuracy
- Handles negation and context
- Performs well on product reviews
- Effective for social media text
- Customizable sentiment thresholds

## 🎯 Use Cases

- **Product Review Analysis**: E-commerce sentiment monitoring
- **Social Media Monitoring**: Brand sentiment tracking
- **Customer Feedback**: Service quality assessment
- **Market Research**: Consumer opinion analysis
- **Content Analysis**: Blog post and article sentiment

## 🔍 Advanced Features

### Word Cloud Analysis
- Sentiment-specific word clouds
- Stop word filtering
- Customizable color schemes
- Size-based frequency visualization

### Comparative Analysis
- Side-by-side method comparison
- Correlation analysis between methods
- Method-specific strengths identification
- Ensemble scoring options

### Report Generation
- Automated summary statistics
- Key findings identification
- Most positive/negative examples
- Actionable insights and recommendations

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Additional sentiment analysis methods
- Multi-language support
- Real-time sentiment streaming
- Advanced visualization options

## 📄 License

This project is open source and available under the MIT License.

## 🔗 Future Enhancements

- Deep learning sentiment models
- Emotion detection beyond sentiment
- Aspect-based sentiment analysis
- Real-time dashboard integration
- API endpoint development
