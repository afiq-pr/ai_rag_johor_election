import requests
from bs4 import BeautifulSoup
import time
import json
import pandas as pd
from datetime import datetime, timedelta
from chunk_and_embed import build_database


# pretend to be a normal browser to prevent from some sites block python
headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"}

# The Star tag page loads the article list using this url can be found inspect element
list_url = "https://www.thestar.com.my/loadMoreTopicOpenSearch"

# all the state election articles are tagged State Polls 2026 which include other states
tag_name = "State Polls 2026"

# Johor polling day is 11 July 2026 so we take timeline between there
start_date = datetime(2026, 6, 1)
end_date = datetime(2026, 7, 20)


# GET LIST OF LINKS


all_links = []
current_day = start_date

#Loop find get all article regarding state polls from start date to end
while current_day <= end_date:
    day_text = current_day.strftime("%Y-%m-%d")
    print("getting links for", day_text, "...")

    page = 1
    while True:
        data = {
            "arrTags": tag_name,
            "sdate": day_text + " 00:00:00",
            "edate": day_text + " 23:59:59",
            "pageNum": page
        }

        response = requests.post(list_url, data=data, headers=headers)
        result = response.json()

        articles = result["loadMoreArticles"]

        # if no more articles then stop
        if len(articles) == 0:
            break

        for a in articles:
            all_links.append({
                "id": a["article_id"],
                "title": a["title"],
                "link": a["link"],
                "date": a["publish_date"]
            })

        # check if there is still a next page
        if result["buttonLoadMore"] == False:
            break

        page = page + 1
        time.sleep(1)  # dont spam the server

    current_day = current_day + timedelta(days=1)
    time.sleep(1)

print("total links found:", len(all_links))

# remove duplicate links just in case
unique_links = []
seen = []
for item in all_links:
    if item["link"] not in seen:
        seen.append(item["link"])
        unique_links.append(item)

print("total unique links:", len(unique_links))



# OPEN EACH ARTICLE AND GET THE CONTENT

def get_meta(soup, name):
    # function find metadata
    tag = soup.find("meta", attrs={"name": name})
    if tag is None:
        return ""
    return tag.get("content", "").strip()

# scrape all metadata found
def scrape_article(url):
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        print("  failed:", response.status_code)
        return None

    soup = BeautifulSoup(response.text, "html.parser")

    title = get_meta(soup, "content_title")
    publish_date = get_meta(soup, "datePublished")
    section = get_meta(soup, "content_category")
    keywords = get_meta(soup, "keywords")

    # author is sometimes "NA", then try the agency (eg Bernama)
    author = get_meta(soup, "content_byline")
    if author == "" or author == "NA":
        author = get_meta(soup, "content_agency")
    if author == "" or author == "NA":
        author = "The Star"

    # the article text is inside <div id="story-body">
    body = soup.find("div", id="story-body")
    if body is None:
        print("  no story body found")
        return None

    paragraphs = []
    for p in body.find_all("p"):
        text = p.get_text().strip()
        if text != "":
            paragraphs.append(text)

    content = "\n".join(paragraphs)

    article = {
        "title": title,
        "publish_date": publish_date,
        "author": author,
        "section": section,
        "keywords": keywords,
        "url": url,
        "content": content
    }
    return article


johor_articles = []

# Loop through all links and find only news related to Johor
for i, item in enumerate(unique_links):
    print(i + 1, "/", len(unique_links), item["title"])

    try:
        article = scrape_article(item["link"])
    except Exception as e:
        print("  error:", e)
        article = None

    if article is not None:
        # only keep Johor election articles and skip others
        # if got johor name in it keep it
        is_johor = False
        if "johor polls" in article["keywords"].lower():
            is_johor = True
        if "johor" in article["title"].lower():
            is_johor = True

        if is_johor:
            johor_articles.append(article)
        else:
            print("  skip, not johor")

    time.sleep(1)  # be polite

print("total johor articles:", len(johor_articles))


# save the data locally just in case

with open("johor_election_articles.json", "w", encoding="utf-8") as f:
    json.dump(johor_articles, f, ensure_ascii=False, indent=2)

df = pd.DataFrame(johor_articles)
df.to_csv("johor_election_articles.csv", index=False, encoding="utf-8-sig")
print("data saved locally!")


# call another function to chunk + embed + save into chromadb
# all the chunking code is inside chunk_and_embed.py

build_database(johor_articles)

