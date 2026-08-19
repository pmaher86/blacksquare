from __future__ import annotations

import struct
from enum import IntEnum

HEADER_FORMAT = "<H 11sx H 8s 4s 2s H 12s B B H H H"
HEADER_CKSUM_FORMAT = "<BBHHH"
EXTENSION_HEADER_FORMAT = "<4sHH"

MASKSTRING = "ICHEATED"
ACROSSDOWN = b"ACROSS&DOWN\0"
ENCODING = "ISO-8859-1"
ENCODING_UTF8 = "UTF-8"

BLACKSQUARE = "."
BLACKSQUARE2 = ":"
BLANKSQUARE = "-"


class PuzzleFormatError(Exception):
    """Raised when a PUZ file has invalid structure or checksums."""


class PuzzleType(IntEnum):
    Normal = 0x0001
    Diagramless = 0x0401


class SolutionState(IntEnum):
    Unlocked = 0x0000
    NotProvided = 0x0002
    Locked = 0x0004


class GridMarkup(IntEnum):
    Default = 0x00
    PreviouslyIncorrect = 0x10
    Incorrect = 0x20
    Revealed = 0x40
    Circled = 0x80


class ExtensionCode:
    Rebus = b"GRBS"
    RebusSolutions = b"RTBL"
    RebusFill = b"RUSR"
    Timer = b"LTIM"
    Markup = b"GEXT"


def data_cksum(data: bytes, cksum: int = 0) -> int:
    """Computes the 16-bit CRC-like checksum used by Across Lite."""
    for b in data:
        lowbit = cksum & 0x0001
        cksum = cksum >> 1
        if lowbit:
            cksum = cksum | 0x8000
        cksum = (cksum + b) & 0xFFFF
    return cksum


class PuzData:
    """In-memory representation of an Across Lite .puz file."""

    def __init__(self, version: str = "1.3") -> None:
        self.version = version.encode("ascii")
        self.fileversion = version.encode("ascii") + b"\0"
        self.preamble = b""
        self.postscript = b""
        self.title = ""
        self.author = ""
        self.copyright = ""
        self.notes = ""
        self.width = 0
        self.height = 0
        self.solution = ""
        self.fill = ""
        self.clues: list[str] = []
        self.unk1 = b"\0" * 2
        self.unk2 = b"\0" * 12
        self.scrambled_cksum = 0
        self.puzzletype = PuzzleType.Normal
        self.solution_state = SolutionState.Unlocked
        self.extensions: dict[bytes, bytes] = {}
        self._extensions_order: list[bytes] = []

    @property
    def encoding(self) -> str:
        try:
            v = tuple(map(int, self.version.split(b".")))
            if v[0] >= 2:
                return ENCODING_UTF8
        except Exception:
            pass
        return ENCODING

    def encode(self, s: str) -> bytes:
        return s.encode(self.encoding, "replace")

    def decode(self, b: bytes) -> str:
        return b.decode(self.encoding, "replace")

    def header_cksum(self, cksum: int = 0) -> int:
        header_bytes = struct.pack(
            HEADER_CKSUM_FORMAT,
            self.width,
            self.height,
            len(self.clues),
            self.puzzletype,
            self.solution_state,
        )
        return data_cksum(header_bytes, cksum)

    def text_cksum(self, cksum: int = 0) -> int:
        enc = self.encoding
        if self.title:
            cksum = data_cksum(self.title.encode(enc, "replace") + b"\0", cksum)
        if self.author:
            cksum = data_cksum(self.author.encode(enc, "replace") + b"\0", cksum)
        if self.copyright:
            cksum = data_cksum(self.copyright.encode(enc, "replace") + b"\0", cksum)

        for clue in self.clues:
            if clue:
                cksum = data_cksum(clue.encode(enc, "replace"), cksum)

        try:
            v = tuple(map(int, self.version.split(b".")))
            if v >= (1, 3) and self.notes:
                cksum = data_cksum(self.notes.encode(enc, "replace") + b"\0", cksum)
        except Exception:
            if self.notes:
                cksum = data_cksum(self.notes.encode(enc, "replace") + b"\0", cksum)

        return cksum

    def global_cksum(self) -> int:
        enc = self.encoding
        cksum = self.header_cksum()
        cksum = data_cksum(self.solution.encode(enc, "replace"), cksum)
        cksum = data_cksum(self.fill.encode(enc, "replace"), cksum)
        return self.text_cksum(cksum)

    def magic_cksum(self) -> int:
        enc = self.encoding
        cksums = [
            self.header_cksum(),
            data_cksum(self.solution.encode(enc, "replace")),
            data_cksum(self.fill.encode(enc, "replace")),
            self.text_cksum(),
        ]

        cksum_magic = 0
        for i, cksum in enumerate(reversed(cksums)):
            cksum_magic <<= 8
            cksum_magic |= ord(MASKSTRING[len(cksums) - i - 1]) ^ (cksum & 0x00FF)
            cksum_magic |= (
                ord(MASKSTRING[len(cksums) - i - 1 + 4]) ^ (cksum >> 8)
            ) << 32

        return cksum_magic

    @classmethod
    def from_bytes(cls, data: bytes, strict: bool = False) -> PuzData:
        puz = cls()
        idx = data.find(ACROSSDOWN)
        if idx < 2:
            raise PuzzleFormatError(
                "Invalid PUZ file: ACROSS&DOWN magic header not found."
            )

        puz.preamble = data[: idx - 2]
        header_start = idx - 2
        header_size = struct.calcsize(HEADER_FORMAT)
        if len(data) < header_start + header_size:
            raise PuzzleFormatError("PUZ file too short for header.")

        unpacked = struct.unpack_from(HEADER_FORMAT, data, header_start)
        (
            cksum_gbl,
            _across_down,
            cksum_hdr,
            cksum_magic,
            fileversion,
            unk1,
            scrambled_cksum,
            unk2,
            width,
            height,
            numclues,
            puzzletype,
            solution_state,
        ) = unpacked

        puz.fileversion = fileversion
        puz.version = fileversion.rstrip(b"\0")
        puz.unk1 = unk1
        puz.scrambled_cksum = scrambled_cksum
        puz.unk2 = unk2
        puz.width = width
        puz.height = height
        puz.puzzletype = PuzzleType(puzzletype)
        puz.solution_state = SolutionState(solution_state)

        pos = header_start + header_size
        enc = puz.encoding
        n_cells = width * height

        if len(data) < pos + 2 * n_cells:
            raise PuzzleFormatError("PUZ file truncated in solution/fill grids.")

        puz.solution = data[pos : pos + n_cells].decode(enc, "replace")
        pos += n_cells
        puz.fill = data[pos : pos + n_cells].decode(enc, "replace")
        pos += n_cells

        def read_null_str() -> str:
            nonlocal pos
            end = data.find(b"\0", pos)
            if end == -1:
                res = data[pos:].decode(enc, "replace")
                pos = len(data)
                return res
            res = data[pos:end].decode(enc, "replace")
            pos = end + 1
            return res

        puz.title = read_null_str()
        puz.author = read_null_str()
        puz.copyright = read_null_str()

        puz.clues = [read_null_str() for _ in range(numclues)]
        puz.notes = read_null_str()

        # Parse extension sections
        ext_cksum: dict[bytes, int] = {}
        ext_header_size = struct.calcsize(EXTENSION_HEADER_FORMAT)
        while pos + ext_header_size <= len(data):
            code, length, c_ext = struct.unpack_from(EXTENSION_HEADER_FORMAT, data, pos)
            pos += ext_header_size
            if len(data) < pos + length:
                break
            ext_data = data[pos : pos + length]
            pos += length
            if pos < len(data) and data[pos : pos + 1] == b"\0":
                pos += 1
            puz.extensions[code] = ext_data
            ext_cksum[code] = c_ext
            puz._extensions_order.append(code)

        if pos < len(data):
            puz.postscript = data[pos:]

        if strict:
            if cksum_gbl != puz.global_cksum():
                raise PuzzleFormatError("Global checksum mismatch in PUZ file.")
            if cksum_hdr != puz.header_cksum():
                raise PuzzleFormatError("Header checksum mismatch in PUZ file.")
            magic_unpacked = struct.unpack("<Q", cksum_magic)[0]
            if magic_unpacked != puz.magic_cksum():
                raise PuzzleFormatError("Magic checksum mismatch in PUZ file.")
            for code, expected in ext_cksum.items():
                if expected != data_cksum(puz.extensions[code]):
                    raise PuzzleFormatError(
                        f"Extension {code!r} checksum mismatch in PUZ file."
                    )

        return puz

    def to_bytes(self) -> bytes:
        enc = self.encoding
        out = bytearray()
        out.extend(self.preamble)

        # Pack 52-byte header
        magic_bytes = struct.pack("<Q", self.magic_cksum())  # 8 bytes magic cksum
        header = struct.pack(
            HEADER_FORMAT,
            self.global_cksum(),
            b"ACROSS&DOWN",
            self.header_cksum(),
            magic_bytes,
            self.fileversion,
            self.unk1,
            self.scrambled_cksum,
            self.unk2,
            self.width,
            self.height,
            len(self.clues),
            int(self.puzzletype),
            int(self.solution_state),
        )
        out.extend(header)

        out.extend(self.solution.encode(enc, "replace"))
        out.extend(self.fill.encode(enc, "replace"))

        def write_null_str(s: str) -> None:
            out.extend(s.encode(enc, "replace") + b"\0")

        write_null_str(self.title)
        write_null_str(self.author)
        write_null_str(self.copyright)

        for clue in self.clues:
            write_null_str(clue)

        write_null_str(self.notes)

        # Write extensions
        for code, ext_data in self.extensions.items():
            ext_header = struct.pack(
                EXTENSION_HEADER_FORMAT,
                code,
                len(ext_data),
                data_cksum(ext_data),
            )
            out.extend(ext_header)
            out.extend(ext_data)
            out.extend(b"\0")

        out.extend(self.postscript)
        return bytes(out)


def parse_rebus_table(raw_rtbl: str) -> dict[int, str]:
    """Parses an RTBL extension string like ' 0:HEART; 1:SUN/MOON;' into a dict."""
    res: dict[int, str] = {}
    for part in raw_rtbl.split(";"):
        if ":" in part:
            k, v = part.split(":", 1)
            try:
                res[int(k.strip())] = v.strip()
            except ValueError:
                continue
    return res


def serialize_rebus_table(solutions: dict[int, str]) -> str:
    """Serializes a solution dictionary into an Across Lite RTBL string."""
    return ";".join(f"{k:>2}:{v}" for k, v in sorted(solutions.items())) + ";"
