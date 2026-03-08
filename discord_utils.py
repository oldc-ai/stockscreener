from __future__ import annotations

import os
import time
from datetime import datetime

import requests


def chunk_text(text: str, limit: int = 3800) -> list[str]:
    if len(text) <= limit:
        return [text]

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for line in text.splitlines():
        extra = len(line) + 1
        if current and current_len + extra > limit:
            chunks.append("\n".join(current))
            current = [line]
            current_len = len(line)
        else:
            current.append(line)
            current_len += extra

    if current:
        chunks.append("\n".join(current))

    return chunks


def send_to_discord(
    content: str,
    title: str,
    webhook_url: str | None = None,
    color: int = 0x2ECC71,
) -> None:
    webhook = webhook_url or os.getenv("DISCORD_WEBHOOK_URL")
    if not webhook:
        raise ValueError("DISCORD_WEBHOOK_URL is required to send Discord alerts.")

    parts = chunk_text(content)
    for index, part in enumerate(parts, start=1):
        payload = {
            "embeds": [
                {
                    "title": f"{title} ({index}/{len(parts)})" if len(parts) > 1 else title,
                    "description": part,
                    "color": color,
                    "timestamp": datetime.utcnow().isoformat(),
                }
            ]
        }
        response = requests.post(webhook, json=payload, timeout=30)
        response.raise_for_status()
        if index < len(parts):
            time.sleep(1)
