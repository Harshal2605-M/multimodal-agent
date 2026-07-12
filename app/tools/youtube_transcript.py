from collections.abc import Callable

from youtube_transcript_api import (
    NoTranscriptFound,
    YouTubeTranscriptApi,
)
from app.agent.schemas import (
    ToolName,
    ToolResult,
    ToolStatus,
)
from app.models.input import URLType
from app.utils.url_detection import detect_urls
from app.tools.base import AgentTool, ToolInput


TranscriptFetcher = Callable[[str], str]


def fetch_youtube_transcript(
    video_id: str,
) -> str:
    """
    Fetch and normalize transcript text for one YouTube video.

    Prefer an English transcript when available. Otherwise fall back
    to the first available transcript, including auto-generated ones.
    """

    api = YouTubeTranscriptApi()

    transcript_list = api.list(video_id)

    try:
        transcript = transcript_list.find_transcript(
            ["en"]
        )
    except NoTranscriptFound:
        transcript = next(iter(transcript_list))

    fetched_transcript = transcript.fetch()

    return " ".join(
        snippet.text.strip()
        for snippet in fetched_transcript
        if snippet.text.strip()
    )


class YouTubeTranscriptTool(AgentTool):
    """
    Retrieve transcript text for one validated YouTube URL.
    """

    def __init__(
        self,
        *,
        transcript_fetcher: TranscriptFetcher = (
            fetch_youtube_transcript
        ),
    ) -> None:
        self._transcript_fetcher = transcript_fetcher

    @property
    def name(self) -> ToolName:
        return ToolName.YOUTUBE_TRANSCRIPT

    def run(
        self,
        tool_input: ToolInput,
    ) -> ToolResult:
        if len(tool_input.urls) != 1:
            return ToolResult(
                step_id=tool_input.step_id,
                tool_name=self.name,
                status=ToolStatus.FAILED,
                error_code="invalid_url_input",
                error_message=(
                    "YouTube transcript tool requires "
                    "exactly one validated YouTube URL."
                ),
            )

        detected_urls = detect_urls(
            tool_input.urls[0]
        )

        if len(detected_urls) != 1:
            return ToolResult(
                step_id=tool_input.step_id,
                tool_name=self.name,
                status=ToolStatus.FAILED,
                error_code="invalid_youtube_url",
                error_message=(
                    "YouTube transcript tool requires "
                    "a validated YouTube URL."
                ),
            )

        detected_url = detected_urls[0]

        if (
            detected_url.url_type is not URLType.YOUTUBE
            or detected_url.video_id is None
        ):
            return ToolResult(
                step_id=tool_input.step_id,
                tool_name=self.name,
                status=ToolStatus.FAILED,
                error_code="invalid_youtube_url",
                error_message=(
                    "YouTube transcript tool requires "
                    "a validated YouTube URL."
                ),
            )

        try:
            transcript = self._transcript_fetcher(
                detected_url.video_id
            )
        except Exception as exc:
            print(
                "YouTube transcript fetch failed: "
                f"{type(exc).__name__}: {exc}"
            )

            return ToolResult(
                step_id=tool_input.step_id,
                tool_name=self.name,
                status=ToolStatus.FAILED,
                error_code="transcript_fetch_failed",
                error_message=(
                    "YouTube transcript could not be retrieved."
                ),
            )

        normalized_transcript = transcript.strip()

        if not normalized_transcript:
            return ToolResult(
                step_id=tool_input.step_id,
                tool_name=self.name,
                status=ToolStatus.FAILED,
                error_code="empty_transcript",
                error_message=(
                    "YouTube transcript contained no usable text."
                ),
            )

        return ToolResult(
            step_id=tool_input.step_id,
            tool_name=self.name,
            status=ToolStatus.SUCCESS,
            output=normalized_transcript,
            metadata={
                "video_id": detected_url.video_id,
            },
        )