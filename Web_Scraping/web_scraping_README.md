# Web Scraping Project

A Python web scraper that extracts quotes from quotes.toscrape.com to demonstrate web scraping techniques and data collection.

## 📋 Project Overview

This project demonstrates how to:
- Make HTTP requests to websites
- Parse HTML content using BeautifulSoup
- Handle multiple pages of data
- Implement respectful scraping practices
- Save scraped data to CSV format

## 🚀 Features

- **Respectful Scraping**: Includes delays between requests
- **Error Handling**: Robust error handling for network issues
- **Data Export**: Saves data to CSV format
- **Pagination Support**: Handles multiple pages automatically
- **Clean Data Structure**: Organized output with quotes, authors, and tags

## 📁 Project Structure

```
web-scraping-project/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   └── web_scraper.py
├── notebooks/
│   └── web_scraping_demo.ipynb
└── data/
    └── scraped_quotes.csv
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
from src.web_scraper import WebScraper

# Initialize scraper
scraper = WebScraper("http://quotes.toscrape.com", delay=1.0)

# Scrape quotes
quotes = scraper.scrape_quotes()

# Save to CSV
scraper.save_to_csv(quotes, 'quotes.csv')
```

### Run the Complete Script

```bash
python src/web_scraper.py
```

## 📊 Sample Output

The scraper collects the following data for each quote:
- **Text**: The actual quote
- **Author**: Who said it
- **Tags**: Associated categories/themes

Sample CSV output:
```
text,author,tags
"The world as we have created it is a process of our thinking...",Albert Einstein,"change,deep-thoughts,thinking,world"
"It is our choices, Harry, that show what we truly are...",J.K. Rowling,"abilities,choices"
```

## ⚡ Key Features

- **Rate Limiting**: 1-second delay between requests to be respectful
- **User Agent**: Proper headers to avoid being blocked
- **Session Management**: Efficient connection reuse
- **Data Validation**: Handles missing data gracefully
- **Progress Tracking**: Shows scraping progress in real-time

## 🔧 Technical Details

- **Language**: Python 3.7+
- **Main Libraries**: requests, BeautifulSoup4, pandas
- **Target Site**: quotes.toscrape.com (designed for scraping practice)
- **Output Format**: CSV file with UTF-8 encoding

## 📈 Results

This scraper typically collects:
- 100+ quotes from multiple authors
- Associated tags for categorization
- Clean, structured data ready for analysis

## 🤝 Contributing

Feel free to fork this project and submit pull requests for improvements!

## 📄 License

This project is open source and available under the MIT License.

## ⚠️ Disclaimer

This scraper is designed for educational purposes. Always respect robots.txt and website terms of service when scraping.
