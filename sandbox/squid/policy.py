"""Web-policy engine for the MITM proxy — pure logic, no sockets.

This module is imported both by the in-proxy ICAP server (icap_server.py) and by
the host-side unit test (test_policy.py), so it must stay dependency-light
(PyYAML + stdlib) and free of any I/O beyond reading the policy file.

A policy is a list of first-match-wins rules over each request, plus a default:

  * request phase  (REQMOD): `set-header` rules are NON-terminal (their header
    ops accumulate); the first matching `allow`/`block`/`redirect` rule is
    terminal. If none matches, `default:` (allow|deny) decides.
  * response phase (RESPMOD): every matching `rewrite-body` rule applies its
    pattern/replace list to the (already-decompressed, decoded) body.

Matching keys (all AND'd within a rule):
    host          fnmatch globs against the request Host (case-insensitive)
    url / path    regex against the full URL / the path
    method        membership in a list (case-insensitive)
    content_type  RESPMOD only: fnmatch globs against the response Content-Type
"""

from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass, field

import yaml


@dataclass
class Req:
    """The request facts a rule matches against."""
    method: str
    scheme: str          # "http" | "https" (conveyed by Squid via adaptation_meta)
    host: str
    path: str
    headers: dict = field(default_factory=dict)   # lower-cased header name -> value

    @property
    def url(self) -> str:
        return f"{self.scheme}://{self.host}{self.path}"


@dataclass
class Rewrite:
    """An in-place request rewrite of the path (REQMOD).

    Regex-rewrites the request path in place ($1.. backrefs) — e.g. pin a
    registry to an approved path prefix. Same-origin only: rewriting to a
    different HOST does not reroute an intercepted TLS connection (Squid pins the
    upstream to the original server), so cross-host offload uses `redirect`."""
    path_pattern: object = None   # compiled regex or None
    path_replace: str = ""

    def apply_path(self, path: str) -> str:
        if self.path_pattern is None:
            return path
        # translate $1.. backrefs to Python's \1.. for re.sub
        repl = re.sub(r"\$(\d+)", r"\\\1", self.path_replace)
        return self.path_pattern.sub(repl, path)


@dataclass
class Decision:
    """Outcome of the request phase.

    `header_ops` and `rewrite` are applied to the forwarded request only when
    action == allow.
    """
    action: str                              # "allow" | "block" | "redirect"
    header_ops: dict = field(default_factory=lambda: {"set": {}, "remove": []})
    status: int = 403
    message: str = ""
    location: str = ""
    rewrite: object = None                    # Rewrite | None


def _as_list(v):
    if v is None:
        return []
    return v if isinstance(v, list) else [v]


class Rule:
    def __init__(self, raw: dict):
        self.name = str(raw.get("name", "unnamed"))
        self.action = str(raw.get("action", "allow"))
        m = raw.get("match", {}) or {}
        self.host_globs = [h.lower() for h in _as_list(m.get("host"))]
        self.methods = [x.upper() for x in _as_list(m.get("method"))]
        self.ct_globs = [c.lower() for c in _as_list(m.get("content_type"))]
        self._url_re = re.compile(m["url"]) if m.get("url") else None
        self._path_re = re.compile(m["path"]) if m.get("path") else None
        # block params
        self.status = int(raw.get("status", 403))
        self.message = str(raw.get("message", "Blocked by Gemma4 web policy."))
        # redirect param (may contain $1.. backrefs into the url match)
        self.location = raw.get("location", "")
        # set-header params (also merged in by a `rewrite` rule, e.g. to inject
        # credentials for the mirror the request is offloaded to)
        rh = raw.get("request_headers", {}) or {}
        self.set_headers = {str(k): str(v) for k, v in (rh.get("set", {}) or {}).items()}
        self.remove_headers = [str(x) for x in _as_list(rh.get("remove"))]
        # rewrite params: regex-rewrite the request path in place
        self.rewrite = None
        rw = raw.get("rewrite")
        if rw:
            ps = rw.get("path") or {}
            self.rewrite = Rewrite(
                path_pattern=re.compile(ps["pattern"]) if ps.get("pattern") else None,
                path_replace=str(ps.get("replace", "")),
            )
        # rewrite-body params: list of {pattern, replace}
        self.body_subs = []
        for sub in _as_list(raw.get("response_body")):
            self.body_subs.append((re.compile(sub["pattern"]), sub.get("replace", "")))

    # -- matching ---------------------------------------------------------

    def _match_host(self, host: str) -> bool:
        if not self.host_globs:
            return True
        host = host.lower()
        return any(fnmatch.fnmatch(host, g) for g in self.host_globs)

    def matches_request(self, req: Req):
        """Return the url-regex match object (truthy) or True/False.

        The returned match object (when the rule uses `url`) carries the groups
        a redirect `location` expands with $1.. backrefs.
        """
        if not self._match_host(req.host):
            return False
        if self.methods and req.method.upper() not in self.methods:
            return False
        url_m = True
        if self._url_re is not None:
            url_m = self._url_re.search(req.url)
            if not url_m:
                return False
        if self._path_re is not None and not self._path_re.search(req.path):
            return False
        return url_m

    def matches_response(self, req: Req, content_type: str) -> bool:
        if not self.matches_request(req):
            return False
        if self.ct_globs:
            ct = (content_type or "").split(";")[0].strip().lower()
            if not any(fnmatch.fnmatch(ct, g) for g in self.ct_globs):
                return False
        return True

    def expand_location(self, url_match) -> str:
        """Expand $1.. in the redirect location using the url-regex groups."""
        if not isinstance(url_match, re.Match):
            return self.location

        def repl(m):
            idx = int(m.group(1))
            try:
                return url_match.group(idx) or ""
            except (IndexError, re.error):
                return ""

        return re.sub(r"\$(\d+)", repl, self.location)


class Policy:
    def __init__(self, raw_text: str, doc: dict):
        self.raw = raw_text
        self.default = str((doc or {}).get("default", "allow")).lower()
        self.rules = [Rule(r) for r in ((doc or {}).get("rules") or [])]

    # -- request phase ----------------------------------------------------

    def decide_request(self, req: Req) -> Decision:
        header_ops = {"set": {}, "remove": []}
        for rule in self.rules:
            m = rule.matches_request(req)
            if not m:
                continue
            if rule.action == "set-header":
                header_ops["set"].update(rule.set_headers)
                header_ops["remove"].extend(rule.remove_headers)
                continue
            if rule.action == "rewrite":
                # Terminal: forward the request with the URL rewrite plus this
                # rule's headers and any accumulated from earlier set-header rules.
                header_ops["set"].update(rule.set_headers)
                header_ops["remove"].extend(rule.remove_headers)
                return Decision("allow", header_ops, rewrite=rule.rewrite)
            if rule.action == "allow":
                return Decision("allow", header_ops)
            if rule.action == "block":
                return Decision("block", header_ops,
                                status=rule.status, message=rule.message)
            if rule.action == "redirect":
                return Decision("redirect", header_ops,
                                location=rule.expand_location(m))
            # rewrite-body is response-phase; ignore here
        if self.default == "deny":
            return Decision("block", header_ops, status=403,
                            message="Blocked by Gemma4 web policy (default deny).")
        return Decision("allow", header_ops)

    # -- response phase ---------------------------------------------------

    def decide_response(self, req: Req, res_headers: dict, body: bytes):
        """Return adapted body bytes, or None if nothing changed.

        `res_headers` keys are lower-cased. Only rules whose content_type glob
        matches the response are applied. The body is decoded with the response
        charset (default utf-8); if it can't be decoded it is left untouched.
        """
        content_type = res_headers.get("content-type", "")
        subs = []
        for rule in self.rules:
            if rule.action != "rewrite-body":
                continue
            if rule.matches_response(req, content_type):
                subs.extend(rule.body_subs)
        if not subs:
            return None

        charset = "utf-8"
        m = re.search(r"charset=([\w\-]+)", content_type, re.I)
        if m:
            charset = m.group(1)
        try:
            text = body.decode(charset, errors="strict")
        except (UnicodeDecodeError, LookupError):
            return None  # not safely text — don't risk corrupting it

        new = text
        for pat, repl in subs:
            new = pat.sub(repl, new)
        if new == text:
            return None
        return new.encode(charset, errors="replace")


def load_policy(path: str) -> Policy:
    """Read and parse a policy file. Raises ValueError on malformed content."""
    with open(path, "r", encoding="utf-8") as fh:
        raw_text = fh.read()
    try:
        doc = yaml.safe_load(raw_text) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"policy YAML parse error: {exc}") from exc
    if not isinstance(doc, dict):
        raise ValueError("policy must be a YAML mapping")
    default = str(doc.get("default", "allow")).lower()
    if default not in ("allow", "deny"):
        raise ValueError(f"policy 'default' must be allow|deny, got {default!r}")
    return Policy(raw_text, doc)
