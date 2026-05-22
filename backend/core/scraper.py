import asyncio
import httpx
from bs4 import BeautifulSoup
import urllib.parse
import re
from typing import List, Dict, Any

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
    "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
    "Accept-Encoding": "gzip, deflate",
}

async def fetch_page(client: httpx.AsyncClient, url: str, params: Dict[str, Any] = None) -> str:
    response = await client.get(url, params=params, headers=HEADERS, timeout=10.0)
    response.raise_for_status()
    return response.text

def extract_naver_news_urls(search_html: str) -> List[str]:
    soup = BeautifulSoup(search_html, 'html.parser')
    list_news = soup.find(class_="list_news")
    urls = []
    if list_news:
        links = list_news.find_all("a", href=True)
        for a in links:
            href = a["href"]
            if "news.naver.com" in href:
                urls.append(href)
    return urls

def parse_naver_article(html: str, url: str) -> Dict[str, str]:
    soup = BeautifulSoup(html, 'html.parser')
    
    # Title
    title_el = soup.select_one('.media_end_head_headline')
    title = title_el.text.strip() if title_el else ""
    if not title:
        meta_title = soup.select_one('meta[property="og:title"]')
        title = meta_title.get('content', '').strip() if meta_title else ""
        
    # Date
    date_el = soup.select_one('.media_end_head_info_datestamp_time._ARTICLE_DATE_TIME') or soup.select_one('.media_end_head_info_datestamp_time')
    date = ""
    if date_el:
        date = date_el.get('data-date-time', '').strip()
        if not date:
            date = date_el.text.strip()
    if not date:
        meta_date = soup.select_one('meta[property="article:published_time"]')
        date = meta_date.get('content', '').strip() if meta_date else ""
        
    # Publisher
    logo_img = soup.select_one('.media_end_head_top_logo img')
    publisher = logo_img.get('alt', '').strip() if logo_img else ""
    if not publisher:
        meta_author = soup.select_one('meta[property="og:article:author"]')
        if meta_author:
            author_text = meta_author.get('content', '')
            publisher = author_text.split('|')[0].strip()
            
    # Body
    body_el = soup.select_one('#dic_area')
    body = body_el.get_text(separator="\n").strip() if body_el else ""
    if not body:
        body_el = soup.select_one('#articleBodyContents')
        body = body_el.get_text(separator="\n").strip() if body_el else ""
        
    return {
        "title": title,
        "link": url,
        "date": date,
        "publisher": publisher,
        "body": body
    }

async def scrape_news(stock_name: str, limit: int = 15) -> List[Dict[str, str]]:
    """
    주어진 주식 종목명에 대해 네이버 뉴스 검색을 수행하고,
    검색된 뉴스들 중 네이버 뉴스 링크가 존재하는 기사 본문 및 메타데이터를 비동기로 크롤링합니다.
    """
    search_url = "https://search.naver.com/search.naver"
    
    # 1. 2페이지 분량의 검색 결과 페이지를 비동기로 가져옵니다 (최대 20개 뉴스 노출)
    async with httpx.AsyncClient(follow_redirects=True) as client:
        # 두 페이지 요청을 병렬 처리
        tasks = [
            fetch_page(client, search_url, params={"where": "news", "query": stock_name, "start": start})
            for start in [1, 11]
        ]
        pages = await asyncio.gather(*tasks, return_exceptions=True)
        
        # URL 추출 및 중복 제거
        all_urls = []
        for page in pages:
            if isinstance(page, Exception):
                continue
            all_urls.extend(extract_naver_news_urls(page))
            
        # URL 중복 제거 및 단순화
        unique_urls = []
        seen_keys = set()
        for url in all_urls:
            match = re.search(r'/article/(\d+)/(\d+)', url)
            if match:
                key = match.group(0)
                if key not in seen_keys:
                    seen_keys.add(key)
                    unique_urls.append(url)
            else:
                if url not in seen_keys:
                    seen_keys.add(url)
                    unique_urls.append(url)
                    
        # limit 제한
        target_urls = unique_urls[:limit]
        
        # 2. 각 뉴스 기사 본문 및 메타데이터를 비동기로 크롤링합니다
        article_tasks = [
            fetch_page(client, url) for url in target_urls
        ]
        article_pages = await asyncio.gather(*article_tasks, return_exceptions=True)
        
        results = []
        for url, page in zip(target_urls, article_pages):
            if isinstance(page, Exception):
                continue
            try:
                article_data = parse_naver_article(page, url)
                if article_data["title"] and article_data["body"]:
                    results.append(article_data)
            except Exception:
                continue
                
        return results
