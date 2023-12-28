from playwright.sync_api import sync_playwright
from nested_lookup import nested_lookup


def scrape_tweet(url: str) -> dict:
    """
    Scrape a single tweet page for Tweet thread e.g.:
    https://twitter.com/Scrapfly_dev/status/1667013143904567296
    Return parent tweet, reply tweets and recommended tweets
    """
    _xhr_calls = []

    def intercept_response(response):
        """capture all background requests and save them"""
        # we can extract details from background requests
        if response.request.resource_type == "xhr":
            _xhr_calls.append(response)
        return response

    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        context = browser.new_context(viewport={"width": 1920, "height": 1080})
        page = context.new_page()

        # enable background request intercepting:
        page.on("response", intercept_response)
        # go to url and wait for the page to load
        page.goto(url, wait_until="networkidle")
        #page.wait_for_timeout(10000)
        #page.wait_for_selector(['css-1dbjc4n'])
        page.wait_for_selector("[data-testid='tweet']")

        # find all tweet background requests:
        #tweet_calls = [f for f in _xhr_calls if "TweetDetail" in f.url]
        tweet_calls = [f for f in _xhr_calls if "UserTweets" in f.url]
        tweets = []
        for xhr in tweet_calls:
            data = xhr.json()
            xhr_tweets = nested_lookup("tweet_results", data)
            tweets.extend([tweet["result"] for tweet in xhr_tweets])

        # Now that we have all tweets we can parse them into a thread
        # The first tweet is the parent, the rest are replies or suggested tweets
        
        #parent = tweets.pop(0)
        replies = []
        other = []
        text = []
        for tweet in tweets:
            text.append(tweet["legacy"]["full_text"])
            #if tweet["conversation_id"] == parent["conversation_id"]:
            #if tweet["rest_id"] == parent["rest_id"]:
                #replies.append(tweet)
            #else:
                #other.append(tweet)
        # using custom sorting to sort by tweets[3]["legacy"]["created_at"] and return the latest tweet's text
        return text
    # {
    #         "tweet": parent,
    #         "replies": replies,
    #         "other": other,  # ads, recommended etc
    #     }


if __name__ == "__main__":
    #print(scrape_tweet("https://twitter.com/Scrapfly_dev/status/1664267318053179398"))
    #print(scrape_tweet("https://twitter.com/illuviumio"))
    #print(scrape_tweet("https://twitter.com/PancakeSwap"))
    #print(scrape_tweet("https://twitter.com/ApeironNFT"))