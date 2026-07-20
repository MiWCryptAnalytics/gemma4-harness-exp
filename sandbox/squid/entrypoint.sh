#!/bin/sh
# Program the transparent-redirect + fail-closed egress rules, then run Squid in
# the foreground. This runs as root (the container's default user) so it can
# touch iptables; Squid drops to the unprivileged `squid` uid on its own.
#
# The sandbox shares THIS container's network namespace (docker run
# --network container:<proxy>), so the agent's connections are locally
# generated here and are matched in the nat OUTPUT chain (not PREROUTING).
set -e

UID_SQUID=3128        # Squid's uid — exempted so its upstream fetches don't loop
HTTP_PORT=3129
HTTPS_PORT=3130

# The image ships an initialized cert db; recreate it if it's somehow absent.
if [ ! -d /var/spool/squid/ssl_db ]; then
    /usr/local/squid/libexec/security_file_certgen -c -s /var/spool/squid/ssl_db -M 4MB
    chown -R squid:squid /var/spool/squid
fi

# --- transparent redirect (nat OUTPUT: locally-generated sandbox traffic) ---
iptables -t nat -A OUTPUT -m owner --uid-owner "$UID_SQUID" -j RETURN   # never redirect Squid's own upstream
iptables -t nat -A OUTPUT -d 127.0.0.0/8 -j RETURN                      # already-local traffic
iptables -t nat -A OUTPUT -p tcp --dport 80  -j REDIRECT --to-ports "$HTTP_PORT"
iptables -t nat -A OUTPUT -p tcp --dport 443 -j REDIRECT --to-ports "$HTTPS_PORT"

# --- fail-closed egress (filter OUTPUT): only proxied HTTP/S + DNS leave -----
# Accept by DESTINATION 127/8, not `-o lo`: REDIRECT rewrites the dst to
# 127.0.0.1 but route_localnet routes it out eth0 (not lo), so an `-o lo` rule
# would miss the redirected packets and the DROP policy would eat them.
iptables -A OUTPUT -d 127.0.0.0/8 -j ACCEPT
iptables -A OUTPUT -o lo -j ACCEPT
iptables -A OUTPUT -m state --state ESTABLISHED,RELATED -j ACCEPT
iptables -A OUTPUT -p udp --dport 53 -j ACCEPT                          # DNS (agent resolves names)
iptables -A OUTPUT -p tcp --dport 53 -j ACCEPT
iptables -A OUTPUT -m owner --uid-owner "$UID_SQUID" -j ACCEPT          # Squid -> real upstream 80/443
iptables -P OUTPUT DROP                                                 # everything else: dropped

# Block IPv6 outright so it can't bypass the IPv4-only intercept above.
ip6tables -P OUTPUT DROP 2>/dev/null || true

# The sandbox agent (uid 1000) shares this netns and could otherwise reach the
# ICAP policy server on 127.0.0.1:1344 over loopback. Reject it by owner-uid
# (legit ICAP traffic is squid uid 3128 only); insert at the top so this wins
# over the loopback ACCEPT appended above. Does NOT affect the redirect.
iptables -I OUTPUT 1 -m owner --uid-owner 1000 -p tcp --dport 1344 -j REJECT --reject-with tcp-reset

# Start the ICAP policy server as the unprivileged squid uid, then wait for it
# to bind 1344 before launching Squid — so Squid's startup OPTIONS probe
# succeeds (with bypass=off a missing ICAP service would fail closed).
runuser -u squid -- env GEMMA_POLICY="${GEMMA_POLICY:-/etc/squid/policy.yaml}" \
    python3 /usr/local/bin/icap_server.py &
for _ in $(seq 1 40); do
    ss -ltn 2>/dev/null | grep -q ':1344' && break
    sleep 0.25
done
ss -ltn 2>/dev/null | grep -q ':1344' || { echo "ICAP server failed to bind 1344" >&2; exit 1; }

# Surface every intercepted request on the container's stdout (Squid can't log
# to /dev/stdout directly once it drops privileges). tail -F waits for Squid to
# create the file. Squid itself (-d1) streams startup/errors to stderr.
tail -F /var/spool/squid/access.log 2>/dev/null &

exec /usr/local/squid/sbin/squid -N -d1 -f /etc/squid/squid.conf
