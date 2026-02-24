import requests
from bs4 import BeautifulSoup

def crawl_website(url):

    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")

    # Remove scripts and style
    for script in soup(["script", "style"]):
        script.decompose()

    text = soup.get_text(separator="\n")

    return text