"""Fetch a URL and render it as readable text — stdlib only.

Staged into the sandbox and run there (like music.py is for compose_music), so
the request leaves through the transparent Squid/ICAP proxy and is subject to the
web policy. No proxy configuration is needed: the sandbox's iptables rules
redirect port 80/443 and its trust store contains only the MITM CA.

Stdlib-only on purpose — the sandbox image has no requests/beautifulsoup, and
`html.parser` is enough to turn a page into readable text.

    python3 webfetch.py URL [QUESTION]
"""

import html
import re
import sys
from html.parser import HTMLParser

MAX_BYTES = 512 * 1024      # cap the download; pages beyond this are truncated
TIMEOUT = 20
MAX_CHARS = 6000            # cap the returned text (context AND terminal output)
USER_AGENT = "Gemma4-Agent/1.0 (+sandboxed harness)"

# Elements whose contents are never readable text.
_DROP = {"script", "style", "noscript", "template", "svg", "canvas"}
# Elements that imply a line break in the rendered text.
_BLOCK = {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6",
          "section", "article", "header", "footer", "blockquote", "pre", "ul",
          "ol", "table", "figure", "hr"}


class _TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.title = ""
        self._drop_depth = 0
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        if tag in _DROP:
            self._drop_depth += 1
        elif tag == "title":
            self._in_title = True
        elif tag in _BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in _DROP:
            self._drop_depth = max(0, self._drop_depth - 1)
        elif tag == "title":
            self._in_title = False
        elif tag in _BLOCK:
            self.parts.append("\n")

    def handle_data(self, data):
        if self._drop_depth:
            return
        if self._in_title:
            self.title += data.strip()
            return
        if data.strip():
            self.parts.append(data)


def extract_text(markup):
    """(title, readable_text) from HTML markup."""
    parser = _TextExtractor()
    try:
        parser.feed(markup)
        parser.close()
    except Exception:
        # Malformed markup: fall back to a crude tag strip rather than failing.
        stripped = re.sub(r"(?is)<(script|style)\b.*?</\1>", " ", markup)
        return "", html.unescape(re.sub(r"(?s)<[^>]+>", " ", stripped))
    text = "".join(parser.parts)
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r" ?\n ?", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return parser.title.strip(), text.strip()


def _stem(word):
    """Crude prefix stem so 'refunds' matches 'refund' and 'limit' 'limits'."""
    return word[:5]


def rank_paragraphs(text, question, limit=MAX_CHARS):
    """Reorder paragraphs by overlap with the question's terms.

    Only used when a question is supplied — for a page much larger than the
    budget it beats returning the first N characters (nav/boilerplate).
    """
    terms = {_stem(w) for w in re.findall(r"\w+", question.lower()) if len(w) > 2}
    paras = [p for p in text.split("\n\n") if p.strip()]
    if not terms or not paras:
        return text[:limit]

    def score(para):
        words = [_stem(w) for w in re.findall(r"\w+", para.lower())]
        return sum(1 for w in words if w in terms)

    scored = sorted(
        ((score(p), -i, p) for i, p in enumerate(paras)),
        reverse=True,
    )
    out, used = [], 0
    for score, _neg_i, para in scored:
        if score == 0:
            continue
        if used + len(para) > limit:
            break
        out.append(para)
        used += len(para)
    if not out:
        return text[:limit]
    # Restore document order so the excerpt reads naturally.
    chosen = set(out)
    return "\n\n".join(p for p in paras if p in chosen)


def fetch(url, max_bytes=MAX_BYTES, timeout=TIMEOUT):
    """Fetch a URL. Returns a dict with status/headers/body (never raises on HTTP).

    A policy-blocked request comes back as an ordinary HTTP error response; that
    body is the operator's block message and is exactly what the agent should
    see, so error statuses are reported rather than swallowed.
    """
    import urllib.error
    import urllib.request

    scheme = url.split(":", 1)[0].lower() if ":" in url else ""
    if scheme not in ("http", "https"):
        return {"error": f"unsupported URL scheme {scheme!r} (only http/https)"}

    req = urllib.request.Request(url, headers={
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,text/plain;q=0.9,*/*;q=0.5",
    })
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
        status, headers, stream = resp.status, resp.headers, resp
    except urllib.error.HTTPError as exc:
        status, headers, stream = exc.code, exc.headers, exc
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}

    with stream:
        raw = stream.read(max_bytes + 1)
    truncated = len(raw) > max_bytes
    return {"status": status, "url": getattr(stream, "url", url),
            "content_type": (headers.get("Content-Type") or "").split(";")[0].strip(),
            "charset": headers.get_content_charset() or "utf-8",
            "body": raw[:max_bytes], "truncated": truncated}


def render(url, question="", max_chars=MAX_CHARS):
    """Fetch and format a page for the model: a short header plus readable text."""
    r = fetch(url)
    if "error" in r:
        return f"Error fetching {url}: {r['error']}"

    ctype = r["content_type"] or "unknown"
    head = [f"URL: {r['url']}", f"Status: {r['status']}", f"Content-Type: {ctype}"]

    if ctype.startswith("image/") or (
            not ctype.startswith("text/")
            and ctype not in ("application/json", "application/xml",
                              "application/xhtml+xml", "unknown")):
        head.append(f"(binary content, {len(r['body'])} bytes — not text)")
        return "\n".join(head)

    try:
        markup = r["body"].decode(r["charset"], errors="replace")
    except LookupError:
        markup = r["body"].decode("utf-8", errors="replace")

    if ctype in ("text/html", "application/xhtml+xml", "unknown"):
        title, text = extract_text(markup)
        if title:
            head.append(f"Title: {title}")
    else:
        text = markup

    if question:
        text = rank_paragraphs(text, question, max_chars)
    if len(text) > max_chars:
        text = text[:max_chars] + f"\n... [truncated at {max_chars} chars]"
    if r["truncated"]:
        head.append(f"(download capped at {MAX_BYTES} bytes)")
    return "\n".join(head) + "\n\n" + text


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: webfetch.py URL [QUESTION]", file=sys.stderr)
        raise SystemExit(2)
    print(render(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else ""))
