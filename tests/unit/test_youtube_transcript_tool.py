from app.agent.schemas import ToolName, ToolStatus
from app.tools.base import ToolInput
from app.tools.youtube_transcript import YouTubeTranscriptTool
from pydantic import SecretStr

from app.config import Settings
from app.tools.youtube_transcript import (
    YouTubeTranscriptTool,
    _build_youtube_api,
)


class FakeTranscriptFetcher:
    def __init__(
        self,
        *,
        transcript: str = "Transcript content.",
        error: Exception | None = None,
    ) -> None:
        self.transcript = transcript
        self.error = error
        self.calls: list[str] = []

    def __call__(
        self,
        video_id: str,
    ) -> str:
        self.calls.append(video_id)

        if self.error is not None:
            raise self.error

        return self.transcript


def test_youtube_transcript_tool_has_authoritative_name() -> None:
    tool = YouTubeTranscriptTool(
        transcript_fetcher=FakeTranscriptFetcher()
    )

    assert tool.name is ToolName.YOUTUBE_TRANSCRIPT


def test_youtube_transcript_tool_returns_transcript() -> None:
    fetcher = FakeTranscriptFetcher(
        transcript="  Transcript content.  "
    )

    tool = YouTubeTranscriptTool(
        transcript_fetcher=fetcher
    )

    result = tool.run(
        ToolInput(
            step_id="step_1",
            query="Get the transcript.",
            urls=[
                "https://www.youtube.com/watch?v=abc123"
            ],
        )
    )

    assert result.status is ToolStatus.SUCCESS
    assert result.output == "Transcript content."
    assert result.metadata == {
        "video_id": "abc123",
    }
    assert fetcher.calls == ["abc123"]


def test_youtube_transcript_tool_rejects_missing_url() -> None:
    fetcher = FakeTranscriptFetcher()

    tool = YouTubeTranscriptTool(
        transcript_fetcher=fetcher
    )

    result = tool.run(
        ToolInput(
            step_id="step_1",
            query="Get the transcript.",
            urls=[],
        )
    )

    assert result.status is ToolStatus.FAILED
    assert result.error_code == "invalid_url_input"
    assert fetcher.calls == []


def test_youtube_transcript_tool_rejects_multiple_urls() -> None:
    fetcher = FakeTranscriptFetcher()

    tool = YouTubeTranscriptTool(
        transcript_fetcher=fetcher
    )

    result = tool.run(
        ToolInput(
            step_id="step_1",
            query="Get the transcript.",
            urls=[
                "https://youtu.be/abc123",
                "https://youtu.be/def456",
            ],
        )
    )

    assert result.status is ToolStatus.FAILED
    assert result.error_code == "invalid_url_input"
    assert fetcher.calls == []


def test_youtube_transcript_tool_rejects_non_youtube_url() -> None:
    fetcher = FakeTranscriptFetcher()

    tool = YouTubeTranscriptTool(
        transcript_fetcher=fetcher
    )

    result = tool.run(
        ToolInput(
            step_id="step_1",
            query="Get the transcript.",
            urls=["https://example.com/video"],
        )
    )

    assert result.status is ToolStatus.FAILED
    assert result.error_code == "invalid_youtube_url"
    assert fetcher.calls == []


def test_youtube_transcript_tool_converts_fetch_failure() -> None:
    fetcher = FakeTranscriptFetcher(
        error=RuntimeError("network failure")
    )

    tool = YouTubeTranscriptTool(
        transcript_fetcher=fetcher
    )

    result = tool.run(
        ToolInput(
            step_id="step_1",
            query="Get the transcript.",
            urls=[
                "https://www.youtube.com/watch?v=abc123"
            ],
        )
    )

    assert result.status is ToolStatus.FAILED
    assert result.error_code == "transcript_fetch_failed"
    assert result.output is None
    assert fetcher.calls == ["abc123"]


def test_youtube_transcript_tool_rejects_empty_transcript() -> None:
    fetcher = FakeTranscriptFetcher(
        transcript="   "
    )

    tool = YouTubeTranscriptTool(
        transcript_fetcher=fetcher
    )

    result = tool.run(
        ToolInput(
            step_id="step_1",
            query="Get the transcript.",
            urls=[
                "https://www.youtube.com/watch?v=abc123"
            ],
        )
    )

    assert result.status is ToolStatus.FAILED
    assert result.error_code == "empty_transcript"


def test_build_youtube_api_without_proxy() -> None:
    settings = Settings(
        youtube_proxy_username=None,
        youtube_proxy_password=None,
    )

    api = _build_youtube_api(settings)

    assert api is not None


def test_build_youtube_api_with_proxy() -> None:
    settings = Settings(
        youtube_proxy_username=SecretStr("test-user"),
        youtube_proxy_password=SecretStr("test-password"),
    )

    api = _build_youtube_api(settings)

    assert api is not None

def test_build_youtube_api_configures_proxy(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeYouTubeTranscriptApi:
        def __init__(
            self,
            *,
            proxy_config=None,
        ) -> None:
            captured["proxy_config"] = proxy_config

    monkeypatch.setattr(
        "app.tools.youtube_transcript.YouTubeTranscriptApi",
        FakeYouTubeTranscriptApi,
    )

    settings = Settings(
        youtube_proxy_username=SecretStr("test-user"),
        youtube_proxy_password=SecretStr("test-password"),
    )

    _build_youtube_api(settings)

    proxy_config = captured["proxy_config"]

    assert proxy_config.proxy_username == "test-user"
    assert proxy_config.proxy_password == "test-password"

def test_build_youtube_api_configures_proxy(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeYouTubeTranscriptApi:
        def __init__(
            self,
            *,
            proxy_config=None,
        ) -> None:
            captured["proxy_config"] = proxy_config

    monkeypatch.setattr(
        "app.tools.youtube_transcript.YouTubeTranscriptApi",
        FakeYouTubeTranscriptApi,
    )

    settings = Settings(
        youtube_proxy_username=SecretStr("test-user"),
        youtube_proxy_password=SecretStr("test-password"),
    )

    _build_youtube_api(settings)

    proxy_config = captured["proxy_config"]

    assert proxy_config.proxy_username == "test-user"
    assert proxy_config.proxy_password == "test-password"

