# utils/reddit_utils.py
import praw
from config import REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT

def get_reddit_instance():
    reddit = praw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_CLIENT_SECRET,
        user_agent=REDDIT_USER_AGENT
    )
    return reddit

def fetch_reddit_posts(subreddit, limit=10):
    reddit = get_reddit_instance()
    posts = []
    for post in reddit.subreddit(subreddit).hot(limit=limit):
        content = f"{post.title}\n{post.selftext}"
        posts.append(content)
    return posts
