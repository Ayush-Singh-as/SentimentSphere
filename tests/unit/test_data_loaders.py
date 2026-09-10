"""Loader contracts: exclusions, conflicts, and malformed names must be explicit."""

from __future__ import annotations

from pathlib import Path

import pytest

from sentimentsphere.data.audio import combine_audio, load_audio, parse_audio_name
from sentimentsphere.data.text import clean_text, load_text, text_identity


def touch_wav(root: Path, relative: str, payload: bytes = b"RIFFfake") -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


# --------------------------------------------------------------------------- #
# text
# --------------------------------------------------------------------------- #


def test_clean_text_normalizes_whitespace_and_unicode():
    assert clean_text("  hello  \n world  ") == "hello world"
    assert clean_text("café") == clean_text("café")


def test_text_identity_is_case_insensitive():
    assert text_identity("Hello There") == text_identity("hello there")
    assert text_identity("a") != text_identity("b")


def test_load_text_audit_counts(tiny_text_csv: Path):
    dataset = load_text(tiny_text_csv)
    counts = dataset.audit["counts"]
    assert counts["raw_rows"] == 12
    assert counts["excluded_shame"] == 1
    assert counts["conflicting_groups"] == 1, "identical text with two labels must drop out"
    assert counts["collapsed_duplicate_rows"] == 1, "case-variant duplicate collapses"
    assert counts["empty_text"] == 1
    assert counts["retained_rows"] == len(dataset.samples) == 7


def test_load_text_has_no_duplicate_ids_or_groups(tiny_text_csv: Path):
    dataset = load_text(tiny_text_csv)
    ids = [s.id for s in dataset.samples]
    assert len(ids) == len(set(ids))
    assert len({s.group for s in dataset.samples}) == len(ids)


def test_load_text_is_order_stable(tiny_text_csv: Path):
    assert load_text(tiny_text_csv).fingerprint == load_text(tiny_text_csv).fingerprint


def test_load_text_rejects_wrong_columns(tmp_path: Path):
    path = tmp_path / "bad.csv"
    path.write_text("emotion,text\njoy,hi\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Emotion,Text"):
        load_text(path)


def test_load_text_rejects_unknown_label(tmp_path: Path):
    path = tmp_path / "unknown.csv"
    path.write_text("Emotion,Text\nennui,hello there\n", encoding="utf-8")
    with pytest.raises(KeyError):
        load_text(path)


# --------------------------------------------------------------------------- #
# audio filename parsing
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("corpus", "relative", "expected"),
    [
        ("tess", "OAF_back_angry/OAF_back_angry.wav", ("OAF", "angry")),
        ("tess", "YAF_bite_ps/YAF_bite_ps.wav", ("YAF", "ps")),
        ("ravdess", "Actor_01/03-01-05-01-01-01-12.wav", ("12", "05")),
        ("crema_d", "1001_DFA_ANG_XX.wav", ("1001", "ANG")),
        ("savee", "DC_a01.wav", ("DC", "a")),
        ("savee", "JE/sa15.wav", ("JE", "sa")),
    ],
)
def test_parse_audio_name(corpus, relative, expected):
    assert parse_audio_name(Path(relative), corpus) == expected


def test_tess_typo_resolved_only_inside_its_verified_directory():
    good = Path("OAF_neutral/OA_bite_neutral.wav")
    assert parse_audio_name(good, "tess") == ("OAF", "neutral")
    with pytest.raises(ValueError, match="Unverified TESS actor"):
        parse_audio_name(Path("YAF_neutral/OA_bite_neutral.wav"), "tess")


@pytest.mark.parametrize(
    ("corpus", "relative"),
    [
        ("tess", "OAF_x/OAF_back.wav"),
        ("tess", "ZZZ_back_angry/ZZZ_back_angry.wav"),
        ("ravdess", "Actor_01/03-02-05-01-01-01-12.wav"),  # song, not speech
        ("ravdess", "Actor_01/03-01-05.wav"),
        ("crema_d", "abc_DFA_ANG_XX.wav"),
        ("savee", "ZZ_a01.wav"),
        ("savee", "DC_a.wav"),  # missing utterance number
        ("nope", "x.wav"),
    ],
)
def test_parse_audio_name_rejects_malformed(corpus, relative):
    with pytest.raises(ValueError):
        parse_audio_name(Path(relative), corpus)


def test_load_audio_records_actor_and_checksums(tmp_path: Path):
    root = tmp_path / "tess"
    touch_wav(root, "OAF_back_angry/OAF_back_angry.wav", b"a")
    touch_wav(root, "YAF_bite_ps/YAF_bite_ps.wav", b"b")
    dataset = load_audio(root, "tess")
    assert {s.group for s in dataset.samples} == {"tess:OAF", "tess:YAF"}
    assert len({s.content_hash for s in dataset.samples}) == 2
    assert dataset.audit["actors"] == {"tess:OAF": 1, "tess:YAF": 1}


def test_load_audio_logs_the_oa_correction(tmp_path: Path):
    root = tmp_path / "tess"
    touch_wav(root, "OAF_neutral/OA_bite_neutral.wav", b"a")
    dataset = load_audio(root, "tess")
    assert dataset.audit["corrections"][0]["actor"] == "OAF"


def test_load_audio_requires_files(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="acquire tess first"):
        load_audio(tmp_path / "missing", "tess")


def test_combine_audio_keeps_corpus_prefixed_groups(tmp_path: Path):
    tess = tmp_path / "tess"
    savee = tmp_path / "savee"
    touch_wav(tess, "OAF_back_angry/OAF_back_angry.wav", b"a")
    touch_wav(savee, "DC_a01.wav", b"b")
    combined = combine_audio([load_audio(tess, "tess"), load_audio(savee, "savee")])
    assert {s.group for s in combined.samples} == {"tess:OAF", "savee:DC"}
    assert combined.name == "speech_combined"


def test_combine_audio_requires_input():
    with pytest.raises(ValueError):
        combine_audio([])
