# SPDX-License-Identifier: Apache-2.0
"""Voice resolution and system tags, without weights."""

import tarfile

import pytest
import torch

from sglang_omni.models.personaplex.architecture import TEXT_MARKER_IDS, TEXT_PAD_ID
from sglang_omni.models.personaplex.prompts import (
    decode_text,
    load_voice_prompt,
    resolve_voice_path,
    tokenize_text_prompt,
    wrap_system_tags,
)
from sglang_omni.models.personaplex.timeline import REFERENCE_CACHE_POSITIONS


class _Tokenizer:
    def encode(self, text):
        return [len(word) for word in text.split()]

    def decode(self, ids):
        return " ".join(str(i) for i in ids)


def test_system_tags_wrap_once():
    assert wrap_system_tags("  Be kind. ") == "<system> Be kind. <system>"
    assert wrap_system_tags("<system> x <system>") == "<system> x <system>"
    assert tokenize_text_prompt(_Tokenizer(), "") == []
    assert tokenize_text_prompt(_Tokenizer(), "Be kind") == [8, 2, 4, 8]


def test_decode_text_drops_frame_markers():
    ids = [TEXT_PAD_ID, 42, *sorted(TEXT_MARKER_IDS), 7]
    assert decode_text(_Tokenizer(), ids) == "42 7"
    assert decode_text(_Tokenizer(), [TEXT_PAD_ID]) == ""


def test_voice_name_resolves_inside_the_packaged_archive(tmp_path):
    voices = tmp_path / "voices"
    voices.mkdir()
    saved = {
        "embeddings": torch.randn(5, 1, 1, 8, dtype=torch.bfloat16),
        "cache": torch.full((1, 17, REFERENCE_CACHE_POSITIONS), 4, dtype=torch.long),
    }
    torch.save(saved, voices / "NATF2.pt")
    with tarfile.open(tmp_path / "voices.tgz", "w:gz") as tar:
        tar.add(voices, arcname="voices")
    (voices / "NATF2.pt").unlink()
    voices.rmdir()

    path = resolve_voice_path(tmp_path, "NATF2")
    assert path == tmp_path / "voices" / "NATF2.pt"
    prompt = load_voice_prompt(path, load_audio=None)
    assert prompt.frames == 6
    assert (
        prompt.embeddings.shape == (5, 8) and prompt.embeddings.dtype == torch.float32
    )
    assert prompt.tail_codes.shape == (2, 8)
    with pytest.raises(FileNotFoundError, match="packaged voices"):
        resolve_voice_path(tmp_path, "NOPE")
