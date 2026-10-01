"""
Hybrid web scraper using BeautifulSoup and Newspaper3k
"""

import requests
from bs4 import BeautifulSoup
from newspaper import Article
from fake_useragent import UserAgent


def scrape_with_beautifulsoup(url):
    """scraping with BeautifulSoup"""
    try:
        ua = UserAgent()
        headers = {'User-Agent': ua.random}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'footer', 'header']):
            element.decompose()
        
        # Find main content
        main_content = None
        content_selectors = [
            'article', 'main', {'class': 'content'}, {'class': 'article'}, 
            {'class': 'post'}, {'id': 'content'}, 'body'
        ]
        
        for selector in content_selectors:
            if isinstance(selector, str):
                main_content = soup.find(selector)
            else:
                main_content = soup.find('div', selector)
            if main_content:
                break
        
        if not main_content:
            main_content = soup.body
        
        if main_content:
            # Extract text from paragraphs and headers
            scraped_lines = []
            for element in main_content.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'li']):
                text = element.get_text().strip()
                if text:
                    scraped_lines.append(text)
            
            scraped_data = '\n'.join(scraped_lines)
            
            if scraped_data.strip():
                return scraped_data
                
    except requests.exceptions.RequestException as e:
        raise Exception(f"Error fetching the website: {e}")
    except Exception as e:
        raise Exception(f"Error processing content: {e}")
    
    return None


def scrape_with_newspaper(url, language='en'):
    """scraping with Newspaper3k"""
    try:
        ua = UserAgent()
        article = Article(url, language=language)
        article.config.browser_user_agent = ua.random
        article.config.request_timeout = 10
        
        article.download()
        article.parse()
        
        if article.text.strip():
            return article.text
    except Exception as e:
        raise Exception(f"Newspaper3k error: {e}")
    
    return None


def scrape_website(url):
    """
    Scrape website content - tries BeautifulSoup first, then Newspaper3k
    
    Args:
        url: Website URL to scrape
    
    Returns:
        Text content string
        
    Raises:
        Exception: If both scraping methods fail
    """
    # Try BeautifulSoup first
    # print("Executing scraper")
    try:
        text = scrape_with_beautifulsoup(url)
        if text:
            return text
    except:
        pass
    
    # try Newspaper3k
    try:
        text = scrape_with_newspaper(url, language='en')
        if text:
            return text
    except:
        pass
    
    # Both methods failed - raise exception
    raise Exception(f"Could not scrape content from {url}")


# Testing
if __name__ == "__main__":
    # test_url = "https://www.msn.com/en-us/money/markets/dow-soars-more-than-550-points-as-wall-street-embraces-optimism-that-government-might-reopen-soon/ar-AA1QeVes"
    test_url = "https://www.cnbc.com/2025/02/04/asia-tech-stocks-rise-after-trump-pauses-tariffs-on-china-and-mexico.html"
    
    try:
        text = scrape_website(test_url)
        print(text[:500] if len(text) > 500 else text)
        print("Total Text Size", len(text))
    except Exception as e:
        print(f"Scraping failed: {e}")