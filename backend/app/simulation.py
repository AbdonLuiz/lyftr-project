ALLOWED_STATUSES = {429, 500, 502, 503}


def parse_simulated_failures(header: str | None, enabled: bool) -> dict[str, int]:
    """Parse ``X-Simulate-Failure`` into ``{key_id: status}``.

    The header is ignored unless failure simulation is enabled. ``*`` means every key.
    Unknown statuses are dropped so a bad pair cannot change failover behavior.
    """
    if not enabled or header is None:
        return {}
    parsed: dict[str, int] = {}
    for part in header.split(","):
        item = part.strip()
        if not item or "=" not in item:
            continue
        key_id, raw_status = item.split("=", 1)
        key_id = key_id.strip()
        raw_status = raw_status.strip()
        if not key_id or not raw_status.isdigit():
            continue
        status = int(raw_status)
        if status not in ALLOWED_STATUSES:
            continue
        parsed[key_id] = status
    return parsed


def status_for_key(simulated: dict[str, int], key_id: str) -> int | None:
    if key_id in simulated:
        return simulated[key_id]
    return simulated.get("*")
