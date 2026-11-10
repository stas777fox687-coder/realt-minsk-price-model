"""
Сбор объявлений о продаже квартир в Минске с Realt.by -> apartments.csv

Как пользоваться:
  1. Открой в браузере realt.by, выставь фильтры (Продажа, Квартиры, Минск),
     скопируй адрес страницы со списком.
  2. Разведка (один запрос, ничего не пишет в CSV):
         python scraper.py --url "<адрес>" --probe
     Создаст debug/next_data.json и покажет, какие поля нашлись.
  3. Если сопоставление полей выглядит правильно, запусти сбор:
         python scraper.py --url "<адрес>" --pages 40

Парсер ищет JSON в теге __NEXT_DATA__ (Next.js кладёт туда данные страницы),
поэтому он не зависит от вёрстки. Названия полей угадываются по шаблонам (FIELDS ниже);
если что-то не нашлось, поправь шаблон под реальные ключи из debug/next_data.json.
"""
import argparse
import csv
import json
import random
import re
import time
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import requests

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0 Safari/537.36 student-project",
    "Accept-Language": "ru,en;q=0.8",
}

# колонка -> регулярки по имени ключа (ключи проверены по реальному объявлению Realt.by)
FIELDS = {
    "id":           [r"^code$"],
    "price_usd":    [r"^pricerates\.840$"],
    "area_total":   [r"^areatotal$"],
    "area_living":  [r"^arealiving$"],
    "area_kitchen": [r"^areakitchen$"],
    "rooms":        [r"^rooms$"],
    "floor":        [r"^storey$"],
    "floors_total": [r"^storeys$"],
    "year_built":   [r"^buildingyear$"],
    "house_type":   [r"^housetype$"],       # числовой код типа дома
    "repair_state": [r"^repairstate$"],     # числовой код ремонта
    "balcony_type": [r"^balconytype$"],     # числовой код балкона
    "metro_station": [r"^metrostationname$"],
    "metro_time":   [r"^metrotime$"],       # минут до метро
    "street":       [r"^streetname$"],
    "address":      [r"^address$"],
    "town":         [r"^townname$"],
}
NUMERIC = ("price_usd", "area_total", "area_living", "area_kitchen", "rooms", "floor",
           "floors_total", "year_built", "metro_time")
COLUMNS = list(FIELDS) + ["lat", "lon"]


def with_page(url: str, page: int) -> str:
    p = urlparse(url)
    q = dict(parse_qsl(p.query))
    q["page"] = str(page)
    return urlunparse(p._replace(query=urlencode(q)))


def fetch_next_data(session, url):
    r = session.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    m = re.search(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', r.text, re.S)
    if not m:
        raise RuntimeError("В странице нет __NEXT_DATA__: вёрстка устроена иначе, "
                           "нужен HTML-парсер (скинь страницу, поправим).")
    return json.loads(m.group(1))


def flatten(obj, prefix=""):
    """{'a': {'b': 1}} -> {'a.b': 1}; списки не разворачиваем."""
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(flatten(v, f"{prefix}{k}."))
    else:
        out[prefix[:-1]] = obj
    return out


def looks_like_listing(d: dict) -> bool:
    keys = " ".join(flatten(d)).lower()
    return "price" in keys and ("area" in keys or "room" in keys or "square" in keys)


def find_listings(node):
    """Ищет в JSON самый длинный список словарей, похожих на объявления."""
    best = []
    if isinstance(node, list):
        if len(node) >= 3 and all(isinstance(x, dict) for x in node) \
                and sum(looks_like_listing(x) for x in node) >= len(node) * 0.6:
            best = node
        for x in node:
            cand = find_listings(x)
            if len(cand) > len(best):
                best = cand
    elif isinstance(node, dict):
        for v in node.values():
            cand = find_listings(v)
            if len(cand) > len(best):
                best = cand
    return best


def pick(flat: dict, patterns):
    for pat in patterns:
        for key, val in flat.items():
            if re.search(pat, key.lower()) and val not in (None, "", [], {}):
                return key, val
    return None, None


def to_number(v):
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        m = re.search(r"-?\d+(?:[.,]\d+)?", v.replace(" ", "").replace("\u00a0", ""))
        if m:
            return float(m.group().replace(",", "."))
    return None


def parse_listing(raw: dict):
    flat = flatten(raw)
    row, used = {}, {}
    for col, pats in FIELDS.items():
        key, val = pick(flat, pats)
        used[col] = key
        if col in NUMERIC:
            val = to_number(val)
        elif val is not None:
            val = str(val).strip()
        row[col] = val
    loc = raw.get("location")  # [долгота, широта]
    if isinstance(loc, list) and len(loc) == 2:
        row["lon"], row["lat"] = loc
    else:
        row["lon"] = row["lat"] = None
    return row, used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True, help="адрес списка объявлений с фильтрами")
    ap.add_argument("--pages", type=int, default=30)
    ap.add_argument("--probe", action="store_true", help="только разведка первой страницы")
    ap.add_argument("--out", default="apartments.csv")
    args = ap.parse_args()

    session = requests.Session()
    seen, rows = set(), []

    for page in range(1, (2 if args.probe else args.pages + 1)):
        data = fetch_next_data(session, with_page(args.url, page))
        listings = find_listings(data)

        if args.probe:
            Path("debug").mkdir(exist_ok=True)
            Path("debug/next_data.json").write_text(
                json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
            print(f"Найдено объявлений на странице: {len(listings)}")
            if not listings:
                print("Список не найден. Открой debug/next_data.json и найди, где лежат объявления.")
                return
            Path("debug/first_listing.json").write_text(
                json.dumps(listings[0], ensure_ascii=False, indent=1), encoding="utf-8")
            _, used = parse_listing(listings[0])
            print("\nСопоставление колонок с ключами JSON (None = не нашлось):")
            for col, key in used.items():
                print(f"  {col:13} <- {key}")
            print("\nПроверь по debug/first_listing.json и при необходимости поправь FIELDS.")
            return

        new = 0
        for raw in listings:
            row, _ = parse_listing(raw)
            if row["town"] != "Минск":
                continue
            uid = row["id"] or json.dumps(raw, sort_keys=True)[:200]
            if uid in seen:
                continue
            seen.add(uid)
            rows.append(row)
            new += 1
        print(f"стр. {page}: +{new} (всего {len(rows)})")
        if new == 0:
            print("Новых объявлений нет, останавливаюсь.")
            break
        time.sleep(random.uniform(1.0, 2.0))  # вежливая пауза

    with open(args.out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print(f"Сохранено {len(rows)} строк в {args.out}")


if __name__ == "__main__":
    main()
