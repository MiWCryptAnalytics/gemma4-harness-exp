# Transparent MITM proxy for the Gemma 4 sandbox.
#
# The sandbox trusts ONLY our MITM CA, so all of its HTTPS must pass through
# this proxy or fail closed. Squid is compiled FROM SOURCE in a build stage so
# we control the exact feature flags the interception path needs:
#   --with-openssl          TLS + ssl-bump (re-sign each leaf with the MITM CA)
#   --enable-ssl-crtd        builds security_file_certgen (on-the-fly leaf certs)
#   --enable-linux-netfilter intercept mode: recover the original dst via
#                            SO_ORIGINAL_DST for locally-redirected traffic
# The slim runtime stage carries only the compiled Squid + the libs it needs.

# ---- build stage: compile Squid ------------------------------------------
FROM debian:12-slim AS build

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential libssl-dev libltdl-dev wget ca-certificates xz-utils \
    && rm -rf /var/lib/apt/lists/*

ARG SQUID_VERSION=7.6
WORKDIR /src
RUN wget -q "https://github.com/squid-cache/squid/releases/download/SQUID_7_6/squid-${SQUID_VERSION}.tar.xz" \
    && tar xf "squid-${SQUID_VERSION}.tar.xz"
WORKDIR /src/squid-${SQUID_VERSION}
RUN ./configure \
        --prefix=/usr/local/squid \
        --with-openssl \
        --enable-ssl-crtd \
        --enable-icap-client \
        --enable-delay-pools \
        --enable-linux-netfilter \
        --disable-arch-native \
        --with-default-user=squid \
    && make -j"$(nproc)" \
    && make install-strip

# ---- runtime stage -------------------------------------------------------
FROM debian:12-slim

# iptables: install the transparent-redirect rules. openssl/ca-certificates:
# the proxy keeps the FULL system trust store so it can verify REAL upstream
# certs (only the *sandbox* is stripped to just our CA). libssl3/libltdl7:
# runtime deps of the compiled Squid.
# python3 + PyYAML (Debian pkg, no pip/build tools) run the ICAP policy server.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libssl3 libltdl7 iptables iproute2 ca-certificates openssl \
        python3 python3-yaml \
    && rm -rf /var/lib/apt/lists/*

# Dedicated low-privilege uid Squid drops to; MUST differ from the sandbox's
# agent uid (1000) so the redirect rules can exempt Squid's own upstream
# traffic by owner-uid without also exempting the agent.
RUN useradd --system --uid 3128 --home-dir /var/spool/squid --shell /usr/sbin/nologin squid

COPY --from=build /usr/local/squid /usr/local/squid

# The signing CA, baked in so Squid can mint per-host leaves. Squid's https_port
# reads the signing cert AND key from one combined PEM (tls-cert=), so we
# concatenate them. Generated on the host by `make mitm-ca`; key stays local.
COPY mitm/ca.crt /etc/squid/ssl/mitm-ca.crt
COPY mitm/ca.key /etc/squid/ssl/mitm-ca.key

COPY squid/squid.conf    /etc/squid/squid.conf
COPY squid/entrypoint.sh /usr/local/bin/entrypoint.sh

# The ICAP policy server + engine, and the baked default (allow-all) policy.
# An operator overrides the policy at runtime with a read-only bind mount over
# /etc/squid/policy.yaml (see sandbox.py --policy-file); it must be world-readable
# because the ICAP server runs as the unprivileged squid uid.
COPY squid/icap_server.py squid/policy.py /usr/local/bin/
COPY squid/policy.default.yaml /etc/squid/policy.yaml

# Combined cert+key PEM for tls-cert=; init the cert cache; own Squid's dirs.
RUN cat /etc/squid/ssl/mitm-ca.crt /etc/squid/ssl/mitm-ca.key > /etc/squid/ssl/mitm-ca.pem \
    && mkdir -p /var/spool/squid /var/log/squid \
    && /usr/local/squid/libexec/security_file_certgen -c -s /var/spool/squid/ssl_db -M 4MB \
    && chown -R squid:squid /etc/squid/ssl /var/spool/squid /var/log/squid /usr/local/squid/var \
    && chmod 640 /etc/squid/ssl/mitm-ca.key /etc/squid/ssl/mitm-ca.pem \
    && chmod +x /usr/local/bin/entrypoint.sh

# Runs as root so the entrypoint can program iptables; Squid itself drops to
# the unprivileged `squid` user (cache_effective_user, via --with-default-user).
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
