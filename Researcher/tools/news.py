import logging
from gnews import GNews


# because gnews is starting its own handler causing double logs printing
logging.getLogger().handlers.clear()

def google_news_tool(topic: str) -> list[dict]:
    google_news = GNews(
        max_results= 20,
        period= '1y'
    )

    articles = google_news.get_news(topic)
    return articles
