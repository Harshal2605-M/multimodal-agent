from app.security.upload_models import DetectedFileType


PDF_SIGNATURE = b"%PDF-"

JPEG_SIGNATURE = b"\xff\xd8\xff"

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

ID3_SIGNATURE = b"ID3"

RIFF_SIGNATURE = b"RIFF"

WAVE_SIGNATURE = b"WAVE"

FTYP_SIGNATURE = b"ftyp"


M4A_BRANDS = {
    b"M4A ",
    b"M4B ",
    b"mp42",
    b"isom",
}


MIN_SIGNATURE_BYTES = 12


def detect_file_type(header: bytes) -> DetectedFileType | None:
    """
    Detect a supported file type using file signature bytes.

    Return None when the bytes do not match a supported format.
    """

    if header.startswith(PDF_SIGNATURE):
        return DetectedFileType.PDF

    if header.startswith(JPEG_SIGNATURE):
        return DetectedFileType.JPEG

    if header.startswith(PNG_SIGNATURE):
        return DetectedFileType.PNG

    if _is_wav(header):
        return DetectedFileType.WAV

    if _is_mp3(header):
        return DetectedFileType.MP3

    if _is_m4a(header):
        return DetectedFileType.M4A

    return None


def _is_wav(header: bytes) -> bool:
    """
    WAV files use a RIFF container with the WAVE form type.
    """

    return (
        len(header) >= MIN_SIGNATURE_BYTES
        and header[0:4] == RIFF_SIGNATURE
        and header[8:12] == WAVE_SIGNATURE
    )


def _is_mp3(header: bytes) -> bool:
    """
    Detect MP3 files using either an ID3 tag or MPEG audio frame sync.
    """

    if header.startswith(ID3_SIGNATURE):
        return True

    if len(header) < 2:
        return False

    first_byte = header[0]
    second_byte = header[1]

    has_frame_sync = (
        first_byte == 0xFF
        and (second_byte & 0xE0) == 0xE0
    )

    if not has_frame_sync:
        return False

    version_bits = (second_byte >> 3) & 0b11

    layer_bits = (second_byte >> 1) & 0b11

    reserved_version = version_bits == 0b01

    reserved_layer = layer_bits == 0b00

    return not reserved_version and not reserved_layer


def _is_m4a(header: bytes) -> bool:
    """
    Detect M4A-compatible ISO Base Media files.

    The first box should be an ftyp box and contain a recognized
    audio-compatible major or compatible brand.
    """

    if len(header) < 12:
        return False

    if header[4:8] != FTYP_SIGNATURE:
        return False

    box_size = int.from_bytes(
        header[0:4],
        byteorder="big",
    )

    if box_size < 16:
        return False

    available_box_size = min(
        box_size,
        len(header),
    )

    ftyp_box = header[8:available_box_size]

    major_brand = ftyp_box[0:4]

    if major_brand in M4A_BRANDS:
        return True

    compatible_brands_start = 8

    compatible_brands = ftyp_box[
        compatible_brands_start:
    ]

    return any(
        compatible_brands[index:index + 4] in M4A_BRANDS
        for index in range(
            0,
            len(compatible_brands) - 3,
            4,
        )
    )