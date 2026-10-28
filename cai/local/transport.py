"""Explicit local/LAN transport without environment proxies or redirects."""
from __future__ import annotations

import ipaddress
import json
import os
import re
import unicodedata
import urllib.error
import urllib.request
from typing import Any
from urllib.parse import urlsplit, urlunsplit


def local_url(url: str, *, network: str = "local") -> str:
    if network not in {"local", "lan"}:
        raise ValueError("network must be local or lan.")
    parts = urlsplit(url)
    host = parts.hostname or ""
    if host == "localhost":
        host = "127.0.0.1"
    try:
        address = ipaddress.ip_address(host)
    except ValueError as exc:
        raise ValueError("Use localhost or a literal IP address for the model server.") from exc
    lan = any(address in ipaddress.ip_network(block) for block in (
        ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16") if address.version == 4
        else ("fc00::/7",)
    ))
    if (
        not (address.is_loopback or network == "lan" and lan)
        or parts.scheme not in {"http", "https"} or "%" in host
        or parts.username or parts.password or parts.query or parts.fragment
    ):
        raise ValueError("Use a loopback server, or --network lan with a private LAN IP. "
                         "Public endpoints, URL credentials, queries and fragments are blocked.")
    authority = f"[{host}]" if address.version == 6 else host
    if parts.port:
        authority += f":{parts.port}"
    return urlunsplit((parts.scheme, authority, parts.path.rstrip("/"), "", ""))


def api_token(variable: str) -> str:
    if not variable:
        return ""
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", variable):
        raise ValueError("api_key_env must be an environment variable name.")
    value = os.environ.get(variable, "")
    if not value:
        raise ValueError(f"Set {variable} to the server API token before running Cai.")
    if any(ord(char) < 33 or ord(char) > 126 for char in value):
        raise ValueError(f"{variable} must contain a token without whitespace/control characters.")
    return value


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req: Any, fp: Any, code: int, msg: str,
                         headers: Any, newurl: str) -> None:
        raise ValueError("Model server redirects are disabled for local-only inference.")


def server_error_detail(body: bytes, token: str = "") -> str:
    """Bounded diagnostics, treated as untrusted text rather than instructions."""
    text = body.decode("utf-8", errors="replace")
    try:
        data = json.loads(text)
    except ValueError:
        data = None
    if isinstance(data, dict):
        error = data.get("error", data.get("message", ""))
        if isinstance(error, dict):
            error = error.get("message", "")
        text = error if isinstance(error, str) else ""
    # Classify before truncation: backend tracebacks may contain the cause at the end.
    lowered = text.lower()
    if any(marker in lowered for marker in (
        "outofmemory", "out of memory", "insufficient memory", "failed to allocate memory"
    )):
        return ("The model server ran out of memory. Free memory on the model computer "
                "and reload the model, or choose a model that fits the available memory. "
                "Increasing Cai's timeout will not resolve this.")
    if token:
        text = text.replace(token, "[redacted]")
    text = "".join(char for char in text
                   if char in "\n\t" or not unicodedata.category(char).startswith("C"))
    text = " ".join(text.split())
    if "channel error" in text.lower():
        return "Model backend reported Channel Error. Check the model server log for the cause."
    return text[:500] + ("…" if len(text) > 500 else "")


def request_json(url: str, payload: dict[str, Any] | None = None,
                 timeout: float = 3, *, network: str = "local",
                 api_key_env: str = "") -> dict[str, Any]:
    url = local_url(url, network=network)
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    token = api_token(api_key_env)
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(
        url, data=None if payload is None else json.dumps(payload).encode(),
        headers=headers,
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    try:
        with opener.open(request, timeout=timeout) as response:
            raw = response.read(16 * 1024 * 1024 + 1)
        if len(raw) > 16 * 1024 * 1024:
            raise ValueError("Model server response exceeded 16 MB.")
        result = json.loads(raw)
    except urllib.error.HTTPError as exc:
        try:
            if exc.code in {401, 403}:
                raise ValueError(f"Model server returned HTTP {exc.code}. Check --api-key-env "
                                 "and the token's permissions in the server settings.") from exc
            try:
                detail = server_error_detail(exc.read(65536), token)
            except (OSError, ValueError):
                detail = ""
            detail = detail or "Check the model server log for the underlying error."
            raise ValueError(f"Model server returned HTTP {exc.code} at {url}. "
                             f"Server diagnostic: {detail}") from exc
        finally:
            exc.close()
    except urllib.error.URLError as exc:
        if isinstance(exc.reason, TimeoutError):
            raise TimeoutError("Model server connection timed out.") from exc
        raise
    if not isinstance(result, dict):
        raise ValueError("Expected a JSON object from the local server.")
    return result
