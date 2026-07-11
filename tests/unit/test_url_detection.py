from app.models.input import URLType
from app.utils.url_detection import detect_urls


def test_detects_youtube_watch_url() -> None:
    result = detect_urls(
        "Watch https://www.youtube.com/watch?v=abc123"
    )

    assert len(result) == 1

    assert result[0].url_type is URLType.YOUTUBE
    assert result[0].video_id == "abc123"


def test_detects_youtube_short_url() -> None:
    result = detect_urls(
        "Watch https://youtu.be/video456"
    )

    assert len(result) == 1
    assert result[0].video_id == "video456"


def test_removes_trailing_punctuation() -> None:
    result = detect_urls(
        "Watch https://youtu.be/video456."
    )

    assert len(result) == 1
    assert result[0].video_id == "video456"


def test_ignores_ordinary_http_url() -> None:
    result = detect_urls(
        "Read https://example.com/report"
    )

    assert result == []


def test_ignores_non_video_youtube_url() -> None:
    result = detect_urls(
        "Visit https://youtube.com/channel/example"
    )

    assert result == []


def test_rejects_fake_youtube_hostname() -> None:
    result = detect_urls(
        "Visit https://evil-youtube.com/watch?v=abc123"
    )

    assert result == []


def test_removes_duplicate_video_ids() -> None:
    result = detect_urls(
        "https://youtube.com/watch?v=abc123 "
        "https://youtu.be/abc123"
    )

    assert len(result) == 1
    assert result[0].video_id == "abc123"


def test_preserves_first_seen_order() -> None:
    result = detect_urls(
        "https://youtu.be/first123 "
        "https://youtu.be/second456"
    )

    assert [
        item.video_id
        for item in result
    ] == [
        "first123",
        "second456",
    ]


def test_preserves_source_id() -> None:
    result = detect_urls(
        "https://youtu.be/abc123",
        source_id="source_pdf",
    )

    assert result[0].source_id == "source_pdf"


def test_returns_empty_list_without_urls() -> None:
    assert detect_urls(
        "No URL exists here."
    ) == []