# -*- coding: utf-8 -*-
"""
浙江省公共资源交易中心 (ggzy.zj.gov.cn) 抓取脚本
抓取首页公告、以及交易信息列表（招标/采购等）
"""
import requests
from urllib.parse import urljoin
from bs4 import BeautifulSoup
import json

BASE_URL = "https://ggzy.zj.gov.cn"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    "Referer": "https://www.zj.gov.cn/",
    "Connection": "keep-alive",
}

_session = None

def get_session():
    global _session
    if _session is None:
        _session = requests.Session()
        _session.headers.update(HEADERS)
    return _session

def fetch_page(url: str, timeout: int = 15) -> requests.Response:
    url = url if url.startswith("http") else urljoin(BASE_URL + "/", url)
    return get_session().get(url, timeout=timeout)

def parse_homepage(html: str):
    """解析首页：提取通知公告、交易信息等列表项，并区分为公告与招标项目"""
    soup = BeautifulSoup(html, "html.parser")
    items = []
    for li in soup.select("li.list-view-cell"):
        a = li.find("a", class_="clearfix", href=True)
        if not a:
            continue
        name_el = a.find("div", class_="item-name")
        time_el = a.find("div", class_="item-time")
        title = (name_el.get_text(strip=True) if name_el else a.get("title") or "").strip()
        date = time_el.get_text(strip=True) if time_el else ""
        href = a.get("href", "")
        link = urljoin(BASE_URL + "/", href) if href else ""
        if not title:
            continue
        # 交易信息公开(jyxxgk) 多为招标/采购项目，xwzx 多为通知公告
        is_bidding = "/jyxxgk/" in (href or link)
        items.append({
            "title": title,
            "date": date,
            "link": link,
            "type": "bidding" if is_bidding else "notice",
        })
    return items

def find_list_pages(html: str):
    """在首页中查找可能指向招标/采购列表的链接"""
    soup = BeautifulSoup(html, "html.parser")
    candidates = []
    for a in soup.find_all("a", href=True):
        href = a.get("href", "")
        text = (a.get_text(strip=True) or "") + " " + (a.get("title") or "")
        if any(k in href or k in text for k in ["zbgg", "jygg", "招标", "采购", "交易", "infolist", "jyxx"]):
            full = urljoin(BASE_URL + "/", href)
            if full.startswith(BASE_URL) and full not in [c["url"] for c in candidates]:
                candidates.append({"url": full, "text": text[:80]})
    return candidates

# 交易信息公开列表页（工程建设-招标公告）
JYXX_LIST_URL = BASE_URL + "/jyxxgk/002001/002001001/infolist.html"

def parse_list_page(html: str):
    """解析列表页：支持 ewb-public-item（交易信息列表）和 list-view-cell（首页风格）"""
    soup = BeautifulSoup(html, "html.parser")
    items = []
    # 交易信息列表页结构：li.ewb-public-item > a + span.ewb-date
    for li in soup.select("li.ewb-public-item"):
        a = li.find("a", href=True)
        date_el = li.find("span", class_="ewb-date")
        title = (a.get("title") or (a.get_text(strip=True) if a else "")).strip()
        date = date_el.get_text(strip=True) if date_el else ""
        href = a.get("href", "") if a else ""
        link = urljoin(BASE_URL + "/", href) if href else ""
        if title:
            items.append({"title": title, "date": date, "link": link})
    if items:
        return items
    # 备用：首页风格 list-view-cell
    for li in soup.select("li.list-view-cell"):
        a = li.find("a", class_="clearfix", href=True) or li.find("a", href=True)
        if not a:
            continue
        name_el = a.find("div", class_="item-name") or a
        time_el = a.find("div", class_="item-time")
        title = (name_el.get_text(strip=True) if name_el else a.get("title") or "").strip()
        date = time_el.get_text(strip=True) if time_el else ""
        href = a.get("href", "")
        link = urljoin(BASE_URL + "/", href) if href else ""
        if title:
            items.append({"title": title, "date": date, "link": link})
    return items

def fetch_bidding_list(list_url: str = JYXX_LIST_URL, max_pages: int = 3):
    """抓取交易信息列表页，解析招标项目列表"""
    all_projects = []
    for page in range(max_pages):
        if page == 0:
            url = list_url
        else:
            url = list_url + ("&pageNo=%d" % (page + 1) if "?" in list_url else "?pageNo=%d" % (page + 1))
        try:
            r = fetch_page(url)
            if not r.ok:
                break
            items = parse_list_page(r.text)
            if not items:
                break
            all_projects.extend(items)
        except Exception as e:
            print("  list page error:", e)
            break
    return all_projects

def main():
    print("请求首页:", BASE_URL + "/")
    try:
        r = fetch_page(BASE_URL + "/")
        r.raise_for_status()
        html = r.text
        print("状态码:", r.status_code, "  长度:", len(html), "字符")

        # 解析首页并区分公告 vs 招标项目
        all_items = parse_homepage(html)
        notices = [x for x in all_items if x.get("type") == "notice"]
        bidding = [x for x in all_items if x.get("type") == "bidding"]
        print("\n首页 通知公告: %d 条 | 招标/交易项目: %d 条" % (len(notices), len(bidding)))
        print("\n招标/交易项目 (前15条):")
        for i, n in enumerate(bidding[:15], 1):
            print("  %d. [%s] %s" % (i, n["date"], n["title"][:55] + ("..." if len(n["title"]) > 55 else "")))
        if len(bidding) > 15:
            print("  ... 共 %d 条" % len(bidding))

        # 保存为 JSON（含公告与招标项目）
        out = {
            "source": BASE_URL,
            "notices": notices,
            "bidding_projects": bidding,
        }
        list_links = find_list_pages(html)
        out["list_page_candidates"] = list_links[:30]
        with open("ggzy_notices.json", "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print("\n已保存到 ggzy_notices.json（含 bidding_projects）")

        if list_links:
            print("\n发现的列表链接 (前3个):")
            for x in list_links[:3]:
                print("  ", x["url"][:72])

        # 抓取交易信息列表页以获取更多招标项目（工程建设-招标公告）
        print("\n抓取交易信息列表页（工程建设-招标公告）...")
        list_projects = fetch_bidding_list(JYXX_LIST_URL, max_pages=3)
        if list_projects:
            print("  列表页获取到 %d 条招标项目，已合并到 bidding_projects" % len(list_projects))
            out["bidding_projects"] = list_projects
            with open("ggzy_notices.json", "w", encoding="utf-8") as f:
                json.dump(out, f, ensure_ascii=False, indent=2)
        else:
            print("  列表页未解析到条目（可能需登录或页面结构不同），已保留首页招标条目。")
    except requests.RequestException as e:
        print("请求失败:", e)
    except Exception as e:
        print("解析异常:", e)
        raise

if __name__ == "__main__":
    main()
