import re
from urllib.parse import parse_qs, urlparse

from app.models.input import DetectedURL, URLType


URL_PATTERN = re.compile(
    r"https?://[^\s<>\"]+",
    re.IGNORECASE,
)

TRAILING_URL_PUNCTUATION = ".,;:!?)]}"


def _clean_detected_url(url: str) -> str:
    return url.rstrip(TRAILING_URL_PUNCTUATION)


def _extract_youtube_video_id(url: str) -> str | None:
    """
    Return a YouTube video ID only for supported YouTube video URLs.
    """

    parsed = urlparse(url)

    hostname = (parsed.hostname or "").lower()

    if hostname in {
        "youtube.com",
        "www.youtube.com",
        "m.youtube.com",
    }:
        if parsed.path != "/watch":
            return None

        video_ids = parse_qs(
            parsed.query
        ).get("v")

        if not video_ids:
            return None

        video_id = video_ids[0].strip()

        return video_id or None

    if hostname == "youtu.be":
        video_id = parsed.path.lstrip("/").split("/")[0]

        return video_id or None

    return None


def detect_urls(
    text: str,
    *,
    source_id: str | None = None,
) -> list[DetectedURL]:
    """
    Detect supported executable URL inputs.

    Detection is deterministic and does not authorize tool execution.

    The current MVP recognizes only YouTube video URLs.
    """

    if not text:
        return []

    detected_urls: list[DetectedURL] = []

    seen_video_ids: set[str] = set()

    for match in URL_PATTERN.finditer(text):
        cleaned_url = _clean_detected_url(
            match.group(0)
        )

        video_id = _extract_youtube_video_id(
            cleaned_url
        )

        if video_id is None:
            continue

        if video_id in seen_video_ids:
            continue

        seen_video_ids.add(video_id)

        detected_urls.append(
            DetectedURL(
                url=cleaned_url,
                url_type=URLType.YOUTUBE,
                source_id=source_id,
                video_id=video_id,
            )
        )

    return detected_urls