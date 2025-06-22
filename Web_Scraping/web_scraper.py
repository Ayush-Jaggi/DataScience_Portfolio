
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import os
from urllib.parse import urljoin, urlparse

class WebScraper:
    def __init__(self, base_url, delay=1.0):
        """
        Initialize the web scraper

        Args:
            base_url (str): The base URL to scrape
            delay (float): Delay between requests to be respectful
        """
        self.base_url = base_url
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })

    def get_page(self, url):
        """
        Fetch a webpage and return BeautifulSoup object

        Args:
            url (str): URL to fetch

        Returns:
            BeautifulSoup: Parsed HTML content
        """
        try:
            response = self.session.get(url)
            response.raise_for_status()
            time.sleep(self.delay)  # Be respectful to the server
            return BeautifulSoup(response.content, 'html.parser')
        except requests.RequestException as e:
            print(f"Error fetching {url}: {e}")
            return None

    def scrape_quotes(self):
        """
        Scrape quotes from quotes.toscrape.com

        Returns:
            list: List of dictionaries containing quote data
        """
        quotes_data = []
        page = 1

        while True:
            url = f"http://quotes.toscrape.com/page/{page}/"
            soup = self.get_page(url)

            if not soup:
                break

            quotes = soup.find_all('div', class_='quote')

            if not quotes:  # No more quotes
                break

            for quote in quotes:
                text = quote.find('span', class_='text').get_text()
                author = quote.find('small', class_='author').get_text()
                tags = [tag.get_text() for tag in quote.find_all('a', class_='tag')]

                quotes_data.append({
                    'text': text,
                    'author': author,
                    'tags': ', '.join(tags)
                })

            print(f"Scraped page {page} - {len(quotes)} quotes found")
            page += 1

            # Limit to first 5 pages for demo
            if page > 5:
                break

        return quotes_data

    def save_to_csv(self, data, filename):
        """
        Save scraped data to CSV file

        Args:
            data (list): List of dictionaries to save
            filename (str): Output filename
        """
        df = pd.DataFrame(data)

        # Create data directory if it doesn't exist
        os.makedirs('data', exist_ok=True)

        filepath = os.path.join('data', filename)
        df.to_csv(filepath, index=False)
        print(f"Data saved to {filepath}")
        return filepath

def main():
    """
    Main function to run the web scraper
    """
    print("Starting web scraping...")

    # Initialize scraper
    scraper = WebScraper("http://quotes.toscrape.com", delay=1.0)

    # Scrape quotes
    quotes = scraper.scrape_quotes()

    # Save to CSV
    if quotes:
        filepath = scraper.save_to_csv(quotes, 'scraped_quotes.csv')
        print(f"Successfully scraped {len(quotes)} quotes!")
        print(f"Data saved to: {filepath}")
    else:
        print("No data was scraped.")

if __name__ == "__main__":
    main()
