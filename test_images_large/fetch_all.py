# -*- coding: utf-8 -*-
import json, os, time, urllib.request, urllib.parse, re

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
UA = "ColoraturaResearch/1.0 (public-domain-art dataset for a personal ML calibration project)"

def api_get(params, retries=4):
    url = "https://commons.wikimedia.org/w/api.php?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(20 * (attempt + 1))
                continue
            raise
    raise RuntimeError("failed after retries: " + url)

def search_candidates(search_term, min_width=700):
    params = {
        "action": "query", "format": "json",
        "generator": "search",
        "gsrsearch": f"{search_term} filetype:bitmap",
        "gsrnamespace": "6", "gsrlimit": "6",
        "prop": "imageinfo", "iiprop": "url|extmetadata|size", "iiurlwidth": "1400",
    }
    data = api_get(params)
    pages = data.get("query", {}).get("pages", {})
    out = []
    for pid, page in pages.items():
        infos = page.get("imageinfo", [])
        if not infos:
            continue
        info = infos[0]
        if info.get("width", 0) < min_width:
            continue
        meta = info.get("extmetadata", {})
        out.append({
            "title": page.get("title", ""),
            "url": info.get("thumburl") or info["url"],
            "full_url": info["url"],
            "categories": meta.get("Categories", {}).get("value", ""),
            "license_short": meta.get("LicenseShortName", {}).get("value", ""),
            "restrictions": meta.get("Restrictions", {}).get("value", ""),
            "artist_html": meta.get("Artist", {}).get("value", ""),
            "object_name": meta.get("ObjectName", {}).get("value", ""),
            "width": info.get("width"), "height": info.get("height"),
        })
    return out

def is_public_domain(c):
    if c["restrictions"]:
        return False
    cats = c["categories"].lower()
    lic = c["license_short"].lower()
    pd_signals = ["public domain", "pd-art", "pd-old", "cc-pd-mark", "pd-us"]
    return any(s in cats for s in pd_signals) or "public domain" in lic

def strip_tags(html):
    return re.sub("<[^>]+>", "", html or "").strip()

def download(url, dest_path):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = resp.read()
    with open(dest_path, "wb") as f:
        f.write(data)
    return len(data)

TASKS = [
    ("אימפרסיוניזם", "impressionism", "Claude Monet water lilies painting"),
    ("אימפרסיוניזם", "impressionism", "Pierre-Auguste Renoir Bal du moulin de la Galette"),
    ("אימפרסיוניזם", "impressionism", "Camille Pissarro boulevard Montmartre painting"),
    ("אימפרסיוניזם", "impressionism", "Edgar Degas ballet dancers painting"),
    ("אימפרסיוניזם", "impressionism", "Alfred Sisley snow Louveciennes painting"),

    ("אקספרסיוניזם", "expressionism", "Edvard Munch The Scream painting"),
    ("אקספרסיוניזם", "expressionism", "Ernst Ludwig Kirchner street scene painting"),
    ("אקספרסיוניזם", "expressionism", "Egon Schiele self portrait painting"),
    ("אקספרסיוניזם", "expressionism", "Emil Nolde painting"),
    ("אקספרסיוניזם", "expressionism", "Wassily Kandinsky Blaue Reiter painting"),

    ("קוביזם", "cubism", "Pablo Picasso cubism painting"),
    ("קוביזם", "cubism", "Georges Braque cubism painting"),
    ("קוביזם", "cubism", "Piet Mondrian composition painting"),
    ("קוביזם", "cubism", "Kazimir Malevich suprematism painting"),
    ("קוביזם", "cubism", "Fernand Leger painting"),

    ("מינימליזם", "minimalism", "Kazimir Malevich Black Square painting"),
    ("מינימליזם", "minimalism", "Piet Mondrian composition red blue yellow painting"),

    ("ריאליזם", "realism", "Gustave Courbet painting"),
    ("ריאליזם", "realism", "Jean-Francois Millet The Gleaners painting"),
    ("ריאליזם", "realism", "Ilya Repin painting"),
    ("ריאליזם", "realism", "Thomas Eakins painting"),
    ("ריאליזם", "realism", "John Singer Sargent portrait painting"),

    ("סוריאליזם", "surrealism", "Giorgio de Chirico metaphysical painting"),
    ("סוריאליזם", "surrealism", "Odilon Redon dream painting"),
    ("סוריאליזם", "surrealism", "Max Ernst early painting"),
    ("סוריאליזם", "surrealism", "Henri Rousseau The Dream painting"),

    ("אבסטרקט-גסטורלי", "abstract_gestural", "Wassily Kandinsky Composition VII painting"),
    ("אבסטרקט-גסטורלי", "abstract_gestural", "Wassily Kandinsky Improvisation painting"),
    ("אבסטרקט-גסטורלי", "abstract_gestural", "Wassily Kandinsky Composition VI painting"),
    ("אבסטרקט-גסטורלי", "abstract_gestural", "Hilma af Klint abstract painting"),
]

def main():
    manifest = {}
    used_titles = set()
    log_lines = []
    for bucket_he, bucket_slug, term in TASKS:
        try:
            cands = search_candidates(term)
        except Exception as e:
            log_lines.append(f"SEARCH FAILED: {term}: {e}")
            time.sleep(10)
            continue
        picked = None
        for c in cands:
            if c["title"] in used_titles:
                continue
            if is_public_domain(c):
                picked = c
                break
        if not picked:
            log_lines.append(f"NO PD MATCH: [{bucket_he}] {term}")
            time.sleep(10)
            continue
        used_titles.add(picked["title"])
        safe_name = re.sub(r"[^a-zA-Z0-9]+", "_", term.lower()).strip("_")
        fname = f"{bucket_slug}__{safe_name}.jpg"
        dest = os.path.join(OUT_DIR, fname)
        try:
            size = download(picked["url"], dest)
        except Exception as e:
            log_lines.append(f"DOWNLOAD FAILED: [{bucket_he}] {term}: {e}")
            time.sleep(10)
            continue
        manifest[fname] = {
            "artist": strip_tags(picked["artist_html"]),
            "title": picked["object_name"] or picked["title"],
            "intended_style_bucket": bucket_he,
            "wikimedia_commons_title": picked["title"],
            "wikimedia_commons_url": picked["url"],
            "license_tag_observed": picked["license_short"] or "(see categories)",
            "categories_observed": picked["categories"][:300],
            "width": picked["width"], "height": picked["height"],
            "bytes": size,
        }
        log_lines.append(f"OK [{bucket_he}] {term} -> {fname} ({size} bytes, {picked['title']})")
        time.sleep(10)

    with open(os.path.join(OUT_DIR, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    with open(os.path.join(OUT_DIR, "fetch_log.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))

    print(f"TOTAL DOWNLOADED: {len(manifest)}")
    from collections import Counter
    counts = Counter(v["intended_style_bucket"] for v in manifest.values())
    for k, v in counts.items():
        print(k, v)

if __name__ == "__main__":
    main()
