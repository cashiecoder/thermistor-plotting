from __future__ import annotations

import re

BIAS_RE = re.compile(
    r"\b(?P<name>VGS|VDS)\s*=\s*(?P<value>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*V\b",
    re.IGNORECASE,
)


def parse_bias_value(label: str, bias_name: str) -> float | None:
    expected = bias_name.upper()
    for match in BIAS_RE.finditer(label):
        if match.group("name").upper() == expected:
            return float(match.group("value"))
    return None
