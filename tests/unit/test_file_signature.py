from app.security.file_signature import detect_file_type
from app.security.upload_models import DetectedFileType


def test_detects_pdf() -> None:
    header = b"%PDF-1.7\n"

    assert detect_file_type(header) is DetectedFileType.PDF


def test_detects_jpeg() -> None:
    header = b"\xff\xd8\xff\xe0\x00\x10JFIF"

    assert detect_file_type(header) is DetectedFileType.JPEG


def test_detects_png() -> None:
    header = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"

    assert detect_file_type(header) is DetectedFileType.PNG


def test_detects_wav() -> None:
    header = (
        b"RIFF"
        + b"\x24\x00\x00\x00"
        + b"WAVE"
        + b"fmt "
    )

    assert detect_file_type(header) is DetectedFileType.WAV


def test_rejects_riff_file_that_is_not_wav() -> None:
    header = (
        b"RIFF"
        + b"\x24\x00\x00\x00"
        + b"AVI "
    )

    assert detect_file_type(header) is None


def test_detects_mp3_with_id3_tag() -> None:
    header = b"ID3\x04\x00\x00\x00\x00\x00\x15"

    assert detect_file_type(header) is DetectedFileType.MP3


def test_detects_mp3_with_valid_frame_sync() -> None:
    header = b"\xff\xfb\x90\x64"

    assert detect_file_type(header) is DetectedFileType.MP3


def test_rejects_invalid_mp3_reserved_version() -> None:
    header = b"\xff\xeb\x90\x64"

    assert detect_file_type(header) is None


def test_rejects_invalid_mp3_reserved_layer() -> None:
    header = b"\xff\xf9\x90\x64"

    assert detect_file_type(header) is None


def test_detects_m4a_major_brand() -> None:
    header = (
        b"\x00\x00\x00\x18"
        + b"ftyp"
        + b"M4A "
        + b"\x00\x00\x00\x00"
        + b"isom"
        + b"mp42"
    )

    assert detect_file_type(header) is DetectedFileType.M4A


def test_detects_m4a_compatible_brand() -> None:
    header = (
        b"\x00\x00\x00\x18"
        + b"ftyp"
        + b"xxxx"
        + b"\x00\x00\x00\x00"
        + b"isom"
        + b"M4A "
    )

    assert detect_file_type(header) is DetectedFileType.M4A


def test_rejects_unknown_iso_base_media_file() -> None:
    header = (
        b"\x00\x00\x00\x18"
        + b"ftyp"
        + b"xxxx"
        + b"\x00\x00\x00\x00"
        + b"yyyy"
        + b"zzzz"
    )

    assert detect_file_type(header) is None


def test_rejects_unknown_binary_file() -> None:
    header = b"MZ\x90\x00random executable bytes"

    assert detect_file_type(header) is None


def test_rejects_empty_bytes() -> None:
    assert detect_file_type(b"") is None


def test_rejects_too_short_riff_header() -> None:
    assert detect_file_type(b"RIFF") is None


def test_rejects_too_short_ftyp_header() -> None:
    assert detect_file_type(b"\x00\x00\x00\x18ftyp") is None