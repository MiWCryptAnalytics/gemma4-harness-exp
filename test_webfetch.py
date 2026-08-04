"""HTML-to-text extraction for the browse tool (no network, no Docker).

webfetch.py runs inside the sandbox, but it is pure stdlib and host-importable,
so its parsing is tested directly here. The live path (policy + audit) is covered
by test_browse_e2e.py.
"""
import webfetch
from webfetch import extract_text, rank_paragraphs, render

PAGE = """<!doctype html>
<html><head>
  <title>  Widget Docs  </title>
  <style>body { color: red; }</style>
  <script>var tracking = "should never appear";</script>
</head>
<body>
  <nav><a href="/">Home</a></nav>
  <h1>Widget API</h1>
  <p>The widget endpoint returns JSON.</p>
  <p>Rate limits are 100 requests per minute.</p>
  <ul><li>GET /widgets</li><li>POST /widgets</li></ul>
  <noscript>Enable JavaScript</noscript>
  <p>Contact support&amp;billing for a quota increase.</p>
</body></html>"""

title, text = extract_text(PAGE)
assert title == "Widget Docs", repr(title)
assert "should never appear" not in text, "script contents must be dropped"
assert "color: red" not in text, "style contents must be dropped"
assert "Enable JavaScript" not in text, "noscript must be dropped"
assert "Rate limits are 100 requests per minute." in text, text
assert "support&billing" in text, "entities must be unescaped"
# Block elements become line breaks rather than running words together.
assert "JSON.\nRate limits" in text or "JSON.\n\nRate limits" in text, repr(text)
print("extract_text (title, scripts/styles dropped, blocks broken):", repr(title))

# Malformed markup degrades to a crude strip instead of raising.
_t, messy = extract_text("<p>unclosed <b>bold <script>x=1</script>")
assert "bold" in messy and "x=1" not in messy, repr(messy)
print("malformed markup -> still returns text: OK")

# --- question focusing keeps the relevant paragraph, in document order ----
long_text = "\n\n".join([
    "Navigation home about contact",
    "The refund window is thirty days from purchase.",
    "Our office is in Berlin.",
    "Refunds are issued to the original payment method.",
])
focused = rank_paragraphs(long_text, "how do refunds work?", limit=200)
assert "refund window" in focused and "Refunds are issued" in focused, focused
assert focused.index("refund window") < focused.index("Refunds are issued"), \
    "excerpt must stay in document order"
print("rank_paragraphs picks refund paragraphs:", repr(focused[:60]))

# No question, or no term overlap -> plain head-of-document truncation.
assert rank_paragraphs(long_text, "", limit=20) == long_text[:20]
assert rank_paragraphs(long_text, "zzz qqq", limit=20) == long_text[:20]
print("no question / no overlap -> head truncation: OK")


# --- render(): header lines, size cap, and non-text handling --------------
class FakeResponse:
    """Stands in for urlopen's return value."""

    def __init__(self, body, status=200, ctype="text/html; charset=utf-8"):
        self.body, self.status, self.url = body, status, "https://example.test/"
        self._ctype = ctype

    @property
    def headers(self):
        outer = self

        class H:
            def get(self, k, default=None):
                return outer._ctype if k.lower() == "content-type" else default

            def get_content_charset(self):
                return "utf-8"
        return H()

    def read(self, n):
        return self.body[:n]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def fake_fetch(url, **kw):
    r = FakeResponse(PAGE.encode())
    return {"status": r.status, "url": r.url, "content_type": "text/html",
            "charset": "utf-8", "body": r.body, "truncated": False}


_real_fetch = webfetch.fetch
webfetch.fetch = fake_fetch
try:
    out = render("https://example.test/", "what are the rate limits?")
    assert out.startswith("URL: https://example.test/\nStatus: 200\n"), out[:80]
    assert "Title: Widget Docs" in out and "Rate limits" in out, out
    print("render() header + body:", repr(out.split("\n\n")[0]))

    # An oversized page is capped so it can't blow up context or the terminal.
    webfetch.fetch = lambda url, **kw: {
        "status": 200, "url": url, "content_type": "text/plain",
        "charset": "utf-8", "body": (b"x" * 50000), "truncated": True}
    out = render("https://example.test/big")
    assert len(out) < webfetch.MAX_CHARS + 500, len(out)
    assert "truncated at" in out and "download capped" in out, out[-200:]
    print(f"oversized page capped to {len(out)} chars: OK")

    # Binary content is summarized, never dumped.
    webfetch.fetch = lambda url, **kw: {
        "status": 200, "url": url, "content_type": "image/png",
        "charset": "utf-8", "body": b"\x89PNG\r\n\x1a\n" * 100, "truncated": False}
    out = render("https://example.test/logo.png")
    assert "binary content" in out and "PNG" not in out, out
    print("binary content summarized, not dumped:", repr(out.splitlines()[-1]))
finally:
    webfetch.fetch = _real_fetch

# --- non-http schemes are refused before any request is made --------------
for bad in ("file:///etc/passwd", "ftp://example.test/x", "gopher://x"):
    r = webfetch.fetch(bad)
    assert "error" in r and "scheme" in r["error"], (bad, r)
print("non-http(s) schemes refused: OK")

print("\nall webfetch tests passed")
