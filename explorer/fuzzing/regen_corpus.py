#!/usr/bin/env python3

"""Regenerates explorer fuzzer corpus files."""

__copyright__ = """
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

import hashlib
from pathlib import Path
from concurrent import futures
import os
import subprocess
import tempfile
from typing import List, Set
from collections.abc import Iterable

_TESTDATA = "explorer/testdata"
_FUZZER_CORPUS = "explorer/fuzzing/fuzzer_corpus"


def _get_files(folder: str, extension: str) -> Set[str]:
    """Gets the list of files with the specified extension."""
    matching_files = set()
    for root, _, files in os.walk(folder):
        for f in files:
            if os.path.splitext(f)[1] == extension:
                matching_files.add(os.path.join(root, f))
    return matching_files


def _carbon_to_proto(carbon_file: str) -> str:
    """Converts carbon file to text proto string."""
    try:
        p = subprocess.run(
            f"bazel-bin/explorer/fuzzing/ast_to_proto {carbon_file}",
            shell=True,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        text_proto = p.stdout.decode("utf-8")
        print(".", end="", flush=True)
        return text_proto
    except subprocess.SubprocessError:
        print("x", end="", flush=True)
        return ""


def _write_corpus_files(text_protos: Iterable[str], corpus_dir: str) -> None:
    """Writes text proto contents to files in corpus directory."""
    for text_proto in text_protos:
        file_name = (
            Path(corpus_dir)
            .joinpath(hashlib.sha1(text_proto.encode("utf-8")).hexdigest())
            .with_suffix(".textproto")
        )
        with open(file_name, "w") as f:
            f.write(text_proto)


def main() -> None:
    os.chdir(os.path.join(os.path.dirname(__file__), "../.."))

    print("Building ast_to_proto...", flush=True)
    subprocess.check_call(
        [
            "bazel",
            "build",
            "//explorer/fuzzing:ast_to_proto",
        ]
    )
    carbon_sources = _get_files(_TESTDATA, ".carbon")
    print(
        f"Converting {len(carbon_sources)} carbon files to proto...",
        flush=True,
    )
    text_protos: List[str] = []
    with futures.ThreadPoolExecutor() as exec:
        all_protos = exec.map(_carbon_to_proto, carbon_sources)
        text_protos.extend(p for p in all_protos if p)

    with tempfile.TemporaryDirectory() as new_corpus_dir:
        print(
            f"\nWriting {len(text_protos)} corpus files to {new_corpus_dir}...",
            flush=True,
        )
        _write_corpus_files(text_protos, new_corpus_dir)

        print("Building explorer_fuzzer...", flush=True)
        subprocess.check_call(
            [
                "bazel",
                "build",
                "--config=fuzzer",
                "//explorer/fuzzing:explorer_fuzzer.full_corpus",
            ]
        )

        print(
            f"Merging interesting inputs into {_FUZZER_CORPUS}...",
            flush=True,
        )
        subprocess.check_call(
            [
                "bazel-bin/explorer/fuzzing/explorer_fuzzer.full_corpus",
                "-merge=1",
                _FUZZER_CORPUS,
                new_corpus_dir,
            ]
        )
    print("All done!", flush=True)


if __name__ == "__main__":
    main()
