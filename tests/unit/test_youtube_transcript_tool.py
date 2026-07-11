from app.agent.schemas import ToolName, ToolStatus
from app.tools.base import ToolInput
from app.tools.youtube_transcript import YouTubeTranscriptTool


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