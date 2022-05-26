#!/usr/bin/env python3

"""Regenerates explorer fuzzer corpus files."""

__copyright__ = """
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

import hashlib
from pathlib import Path
import argparse
from concurrent import futures
import os
import random
import subprocess
import sys
from typing import Set
from collections.abc import Iterable

_TESTDATA = "explorer/testdata"
_FUZZER_CORPUS = "explorer/fuzzing/fuzzer_corpus"


def _parse_args() -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--delete_old",
        action="store_false",
        help="Whether to delete old corpus files.",
    )
    parser.add_argument(
        "--num_files",
        type=int,
        default=32,
        help="The number of fuzzer corpus files to generate.",
    )
    return parser.parse_args(sys.argv[1:])


def _get_files(folder: str, extension: str) -> Set[str]:
    """Gets the list of files with the specified extension."""
    matching_files = set()
    for root, _, files in os.walk(folder):
        for f in files:
            if os.path.splitext(f)[1] == extension:
                matching_files.add(os.path.join(root, f))
    return matching_files


def _delete_corpus_files() -> None:
    """Deletes corpus files."""
    corpus_files = _get_files(_FUZZER_CORPUS, ".textproto")
    print("Deleting {} old corpus file(s).".format(len(corpus_files)))
    for f in corpus_files:
        os.unlink(f)


def _carbon_to_proto(carbon_file: str) -> str:
    """Converts carbon file to text proto string."""
    try:
        p = subprocess.run(
            (
                "bazel-bin/explorer/fuzzing/fuzzverter --mode carbon_to_proto "
                "--input {} --output /dev/stdout"
            ).format(carbon_file),
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


def _write_corpus_files(text_protos: Iterable[str]) -> None:
    """Writes text proto contents to files in corpus directory."""
    for text_proto in text_protos:
        file_name = (
            Path(_FUZZER_CORPUS)
            .joinpath(hashlib.sha1(text_proto.encode("utf-8")).hexdigest())
            .with_suffix(".textproto")
        )
        with open(file_name, "w") as f:
            print("Writing {}: {} byte(s)".format(file_name, len(text_proto)))
            f.write(text_proto)


def main() -> None:
    os.chdir(os.path.join(os.path.dirname(__file__), "../.."))

    print("Building fuzzverter...")
    subprocess.check_call(["bazel", "build", "//explorer/fuzzing:fuzzverter"])

    parsed_args = _parse_args()
    if parsed_args.delete_old:
        _delete_corpus_files()

    carbon_sources = _get_files(_TESTDATA, ".carbon")
    print("Converting {} carbon files to proto".format(len(carbon_sources)))
    with futures.ThreadPoolExecutor() as exec:
        text_protos = exec.map(_carbon_to_proto, carbon_sources)
        non_empty_text_protos = list(filter(lambda p: len(p) > 0, text_protos))
        random.shuffle(non_empty_text_protos)
        selected_text_protos = non_empty_text_protos[0 : parsed_args.num_files]
        print(
            "\nWriting {} corpus files to {}.".format(
                len(selected_text_protos), _FUZZER_CORPUS
            )
        )
        _write_corpus_files(selected_text_protos)


if __name__ == "__main__":
    main()
