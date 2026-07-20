"""Hand-rolled ICAP server (stdlib only) enforcing the Gemma4 web policy.

Squid (built with --enable-icap-client) vectors every intercepted request to
this server at two points:

  REQMOD  (icap://127.0.0.1:1344/reqmod)  — before contacting the origin:
          allow (204/echo) · modify request headers (200) ·
          block/redirect  (200 with an encapsulated HTTP *response* so Squid
          short-circuits and never touches the origin).
  RESPMOD (icap://127.0.0.1:1344/respmod) — after the origin responds:
          rewrite/redact the (decompressed) response body, or 204 unchanged.

Policy comes from policy.py, loaded from $GEMMA_POLICY (default
/etc/squid/policy.yaml) and hot-reloaded on mtime change. A malformed edit keeps
the last-good policy rather than taking the proxy down.

Runs as the unprivileged squid uid, bound to loopback. The entrypoint firewalls
the sandbox agent's uid off this port so the agent can't reach the ICAP service
over the shared network namespace.
"""

import gzip
import hashlib
import os
import socketserver
import sys
import threading
import time
import zlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from policy import Req, load_policy  # noqa: E402

HOST, PORT = "127.0.0.1", 1344
POLICY_PATH = os.environ.get("GEMMA_POLICY", "/etc/squid/policy.yaml")
MAX_BODY = 8 * 1024 * 1024        # skip RESPMOD rewrite above this (avoid OOM)
CRLF = b"\r\n"


def log(msg):
    print(f"[icap] {msg}", flush=True)


# --------------------------------------------------------------------------
# Policy state + hot reload


class State:
    def __init__(self, path):
        self.path = path
        self.lock = threading.Lock()
        self._mtime = 0
        self.policy = None
        self.istag = '"gm4-boot"'
        self._load(initial=True)

    def _load(self, initial=False):
        try:
            pol = load_policy(self.path)
        except Exception as exc:
            if initial or self.policy is None:
                # Never come up trusting nothing-parsed; fall back to allow-all
                # only at boot so the proxy still serves. Log loudly.
                from policy import Policy
                log(f"WARNING: policy load failed ({exc}); using allow-all until fixed")
                self.policy = Policy("version: 1\ndefault: allow\n", {"default": "allow"})
            else:
                log(f"WARNING: policy reload failed ({exc}); keeping last-good policy")
            return
        self.policy = pol
        self.istag = '"gm4-' + hashlib.sha1(pol.raw.encode()).hexdigest()[:16] + '"'

    def maybe_reload(self):
        try:
            mt = os.path.getmtime(self.path)
        except OSError:
            return
        if mt != self._mtime:
            with self.lock:
                self._mtime = mt
                self._load()


STATE = None  # set in main()


# --------------------------------------------------------------------------
# Low-level ICAP wire helpers


def read_header_block(rf):
    """Read bytes up to and including the terminating CRLFCRLF."""
    buf = b""
    while b"\r\n\r\n" not in buf:
        line = rf.readline()
        if not line:
            return buf or None
        buf += line
    return buf


def parse_icap_headers(block):
    """(request_line:str, headers:dict-lowercased) from an ICAP header block."""
    text = block.split(b"\r\n\r\n", 1)[0].decode("latin1")
    lines = text.split("\r\n")
    request_line = lines[0]
    headers = {}
    for ln in lines[1:]:
        if ":" in ln:
            k, v = ln.split(":", 1)
            headers[k.strip().lower()] = v.strip()
    return request_line, headers


def parse_encapsulated(value):
    """'req-hdr=0, res-hdr=137, res-body=296' -> [('req-hdr',0),('res-hdr',137),...]."""
    out = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        name, off = part.split("=")
        out.append((name.strip(), int(off)))
    return out


def read_chunked(rf):
    """Read an ICAP-chunked body. Return (bytes, saw_ieof)."""
    body, ieof = b"", False
    while True:
        size_line = rf.readline()
        if not size_line:
            break
        token = size_line.strip()
        if b";" in token:
            size_hex, ext = token.split(b";", 1)
            if b"ieof" in ext:
                ieof = True
        else:
            size_hex = token
        try:
            n = int(size_hex, 16)
        except ValueError:
            break
        if n == 0:
            rf.readline()  # trailing CRLF after the 0 chunk
            break
        chunk = rf.read(n)
        body += chunk
        rf.readline()      # CRLF after chunk data
    return body, ieof


def chunk_body(b):
    if not b:
        return b"0\r\n\r\n"
    return f"{len(b):x}".encode() + CRLF + b + CRLF + b"0\r\n\r\n"


def read_encapsulated(rf, enc):
    """Read the encapsulated sections in order. Returns dict of section->bytes,
    with '_ieof' for the body's ieof flag."""
    sections = {}
    for i, (name, off) in enumerate(enc):
        if name == "null-body":
            sections["_body_name"] = None
            break
        if name.endswith("-body"):
            body, ieof = read_chunked(rf)
            sections[name] = body
            sections["_body_name"] = name
            sections["_ieof"] = ieof
            break
        length = enc[i + 1][1] - off
        sections[name] = rf.read(length)
    return sections


# --------------------------------------------------------------------------
# HTTP header (de)serialization for the encapsulated messages


def parse_http_message(block):
    """(start_line:str, headers:list[(name,value)]) from raw HTTP header bytes."""
    text = block.decode("latin1")
    parts = text.split("\r\n")
    start = parts[0]
    headers = []
    for ln in parts[1:]:
        if not ln:
            break
        if ":" in ln:
            k, v = ln.split(":", 1)
            headers.append((k.strip(), v.strip()))
    return start, headers


def serialize_http(start_line, headers):
    out = start_line + "\r\n"
    for k, v in headers:
        out += f"{k}: {v}\r\n"
    out += "\r\n"
    return out.encode("latin1")


def header_get(headers, name):
    name = name.lower()
    for k, v in headers:
        if k.lower() == name:
            return v
    return None


def apply_header_ops(headers, ops):
    remove = {r.lower() for r in ops.get("remove", [])}
    setmap = ops.get("set", {})
    setlower = {k.lower() for k in setmap}
    out = [(k, v) for (k, v) in headers
           if k.lower() not in remove and k.lower() not in setlower]
    for k, v in setmap.items():
        out.append((k, v))
    return out


def apply_rewrite(start_line, headers, rw):
    """Regex-rewrite the request path in place (same host). Returns new_start.

    Cross-host reroute is intentionally not supported: Squid pins an intercepted
    bumped connection to the original upstream, so changing the host here would
    only spoof the Host header against the original server. Use redirect for that.
    """
    try:
        method, target, ver = start_line.split(" ", 2)
    except ValueError:
        return start_line
    if target.startswith("http://") or target.startswith("https://"):
        scheme, rest = target.split("://", 1)
        host = rest.split("/", 1)[0]
        path = "/" + rest.split("/", 1)[1] if "/" in rest else "/"
        new_target = f"{scheme}://{host}{rw.apply_path(path)}"
    else:
        new_target = rw.apply_path(target)
    return f"{method} {new_target} {ver}"


# --------------------------------------------------------------------------
# Request/response construction


def http_block_response(status, message):
    body = (f"<html><body><h1>{status} Blocked</h1><p>{message}</p>"
            "<hr><small>gemma4 mitm proxy</small></body></html>").encode("utf-8")
    hdr = (f"HTTP/1.1 {status} Forbidden\r\n"
           "Content-Type: text/html; charset=utf-8\r\n"
           f"Content-Length: {len(body)}\r\n"
           "Connection: close\r\n\r\n").encode("latin1")
    return hdr, body


def http_redirect_response(location):
    hdr = ("HTTP/1.1 302 Found\r\n"
           f"Location: {location}\r\n"
           "Content-Length: 0\r\n"
           "Connection: close\r\n\r\n").encode("latin1")
    return hdr, b""


def build_req(icap_headers, reqhdr_bytes):
    start, headers = parse_http_message(reqhdr_bytes)
    try:
        method, target, _ver = start.split(" ", 2)
    except ValueError:
        method, target = "GET", "/"
    scheme = icap_headers.get("x-gemma-scheme", "https")
    host = header_get(headers, "host") or ""
    path = target
    if target.startswith("http://") or target.startswith("https://"):
        # absolute-form (explicit proxy): split scheme/host/path
        scheme, rest = target.split("://", 1)
        host = rest.split("/", 1)[0]
        path = "/" + rest.split("/", 1)[1] if "/" in rest else "/"
    return Req(method=method, scheme=scheme, host=host, path=path,
               headers={k.lower(): v for k, v in headers}), start, headers


# --------------------------------------------------------------------------
# Content-Encoding handling for RESPMOD


def decode_body(body, encoding):
    """Return (decoded_bytes, recompress_fn) or (None, None) if unsupported."""
    enc = (encoding or "").strip().lower()
    if enc in ("", "identity"):
        return body, (lambda b: b)
    if enc == "gzip":
        return gzip.decompress(body), gzip.compress
    if enc == "deflate":
        return zlib.decompress(body), zlib.compress
    return None, None  # br / zstd — no stdlib codec; don't touch


# --------------------------------------------------------------------------
# ICAP handler


class Handler(socketserver.StreamRequestHandler):
    def handle(self):
        try:
            while True:
                block = read_header_block(self.rfile)
                if not block:
                    return
                request_line, ihdrs = parse_icap_headers(block)
                method = request_line.split(" ", 1)[0].upper()
                STATE.maybe_reload()
                if method == "OPTIONS":
                    self.do_options(request_line)
                elif method == "REQMOD":
                    self.do_reqmod(request_line, ihdrs)
                elif method == "RESPMOD":
                    self.do_respmod(request_line, ihdrs)
                else:
                    return
                if ihdrs.get("connection", "").lower() == "close":
                    return
        except Exception as exc:  # keep the thread from dying; Squid fails closed
            log(f"handler error: {exc!r}")
            return

    # -- OPTIONS ----------------------------------------------------------

    def do_options(self, request_line):
        is_resp = "/respmod" in request_line
        methods = "RESPMOD" if is_resp else "REQMOD"
        preview = "4096" if is_resp else "0"
        resp = (
            "ICAP/1.0 200 OK\r\n"
            f"Methods: {methods}\r\n"
            "Service: gemma4-icap/1.0\r\n"
            f"ISTag: {STATE.istag}\r\n"
            "Max-Connections: 100\r\n"
            "Options-TTL: 3600\r\n"
            "Allow: 204\r\n"
            f"Preview: {preview}\r\n"
            "Transfer-Preview: *\r\n"
            "Encapsulated: null-body=0\r\n\r\n"
        )
        self.wfile.write(resp.encode("latin1"))

    # -- shared -----------------------------------------------------------

    def _flags(self, ihdrs):
        allow204 = "204" in ihdrs.get("allow", "")
        preview = "preview" in ihdrs
        return allow204, preview

    def _send204(self):
        self.wfile.write(b"ICAP/1.0 204 No Content\r\nISTag: " + STATE.istag.encode()
                         + b"\r\nEncapsulated: null-body=0\r\n\r\n")

    def _send_resp_message(self, reshdr, resbody, has_body):
        """Send a 200 that carries an encapsulated HTTP response."""
        if has_body:
            enc = f"res-hdr=0, res-body={len(reshdr)}"
            payload = reshdr + chunk_body(resbody)
        else:
            enc = f"res-hdr=0, null-body={len(reshdr)}"
            payload = reshdr
        head = ("ICAP/1.0 200 OK\r\nISTag: " + STATE.istag
                + f"\r\nEncapsulated: {enc}\r\n\r\n")
        self.wfile.write(head.encode("latin1") + payload)

    def _send_req_message(self, reqhdr, reqbody, has_body):
        """Send a 200 that carries  firewall rules on the an (adapted) encapsulated HTTP request."""
        if has_body:
            enc = f"req-hdr=0, req-body={len(reqhdr)}"
            payload = reqhdr + chunk_body(reqbody)
        else:
            enc = f"req-hdr=0, null-body={len(reqhdr)}"
            payload = reqhdr
        head = ("ICAP/1.0 200 OK\r\nISTag: " + STATE.istag
                + f"\r\nEncapsulated: {enc}\r\n\r\n")
        self.wfile.write(head.encode("latin1") + payload)

    def _complete_body(self, sections, preview):
        """If a preview didn't include the whole body, pull the rest."""
        name = sections.get("_body_name")
        if not name:
            return b"", False
        body = sections.get(name, b"")
        if preview and not sections.get("_ieof"):
            self.wfile.write(b"ICAP/1.0 100 Continue\r\n\r\n")
            rest, _ = read_chunked(self.rfile)
            body += rest
        return body, True

    # -- REQMOD -----------------------------------------------------------

    def do_reqmod(self, request_line, ihdrs):
        allow204, preview = self._flags(ihdrs)
        enc = parse_encapsulated(ihdrs.get("encapsulated", "req-hdr=0, null-body=0"))
        sections = read_encapsulated(self.rfile, enc)
        reqhdr = sections.get("req-hdr", b"")
        req, start, headers = build_req(ihdrs, reqhdr)
        d = STATE.policy.decide_request(req)

        if d.action == "block":
            log(f"REQMOD block {req.url}")
            hdr, body = http_block_response(d.status, d.message)
            return self._send_resp_message(hdr, body, has_body=True)
        if d.action == "redirect":
            log(f"REQMOD redirect {req.url} -> {d.location}")
            hdr, _ = http_redirect_response(d.location)
            return self._send_resp_message(hdr, b"", has_body=False)

        # allow — possibly with a URL/host rewrite and/or header modifications
        has_ops = bool(d.header_ops["set"] or d.header_ops["remove"])
        has_rewrite = d.rewrite is not None
        if not has_ops and not has_rewrite:
            if allow204:
                return self._send204()
            body, has_body = self._complete_body(sections, preview)
            return self._send_req_message(reqhdr, body, has_body)

        new_start, new_headers = start, headers
        if has_rewrite:
            new_start = apply_rewrite(new_start, new_headers, d.rewrite)
        if has_ops:
            new_headers = apply_header_ops(new_headers, d.header_ops)
        new_reqhdr = serialize_http(new_start, new_headers)
        body, has_body = self._complete_body(sections, preview)
        log(f"REQMOD {'rewrite' if has_rewrite else 'modify'} {req.url}")
        return self._send_req_message(new_reqhdr, body, has_body)

    # -- RESPMOD ----------------------------------------------------------

    def do_respmod(self, request_line, ihdrs):
        allow204, preview = self._flags(ihdrs)
        enc = parse_encapsulated(ihdrs.get("encapsulated", "res-hdr=0, null-body=0"))
        sections = read_encapsulated(self.rfile, enc)
        reqhdr = sections.get("req-hdr", b"")
        reshdr = sections.get("res-hdr", b"")
        body, has_body = self._complete_body(sections, preview)

        req, _s, _h = build_req(ihdrs, reqhdr) if reqhdr else (
            Req("GET", "https", "", "/"), "", [])
        start, resheaders = parse_http_message(reshdr) if reshdr else ("", [])
        res_hdr_map = {k.lower(): v for k, v in resheaders}

        # Decide whether we can/should adapt.
        if not has_body or len(body) > MAX_BODY:
            return self._passthrough_resp(allow204, reshdr, body, has_body)
        decoded, recompress = decode_body(body, res_hdr_map.get("content-encoding"))
        if decoded is None:
            return self._passthrough_resp(allow204, reshdr, body, has_body)

        new = STATE.policy.decide_response(req, res_hdr_map, decoded)
        if new is None:
            return self._passthrough_resp(allow204, reshdr, body, has_body)

        out_body = recompress(new)
        # Fix framing: set the new Content-Length, drop chunked Transfer-Encoding.
        new_headers = [(k, v) for (k, v) in resheaders
                       if k.lower() not in ("content-length", "transfer-encoding")]
        new_headers.append(("Content-Length", str(len(out_body))))
        new_reshdr = serialize_http(start, new_headers)
        log(f"RESPMOD rewrite {req.url} ({len(decoded)}->{len(new)} bytes)")
        self._send_resp_message(new_reshdr, out_body, has_body=True)

    def _passthrough_resp(self, allow204, reshdr, body, has_body):
        if allow204:
            return self._send204()
        self._send_resp_message(reshdr, body, has_body=has_body)


# --------------------------------------------------------------------------


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def main():
    global STATE
    STATE = State(POLICY_PATH)
    # Prime the mtime so the first request doesn't needlessly reload.
    try:
        STATE._mtime = os.path.getmtime(POLICY_PATH)
    except OSError:
        pass
    log(f"policy: {POLICY_PATH} (default={STATE.policy.default}, "
        f"rules={len(STATE.policy.rules)}) istag={STATE.istag}")
    srv = Server((HOST, PORT), Handler)
    log(f"listening on {HOST}:{PORT}")
    srv.serve_forever()


if __name__ == "__main__":
    main()
