#!/usr/bin/env python3

"""Generate a file from a template, substituting the provided key/value pairs.

The file format should match Python's `string.Template` substitution rules:
- `$$` for a literal `$`
- `$identifier` for some key `identifier` to be substituted
- `${identifier}` when adjacent text would be interpreted as part of the
  identifier.

The keys must be strings that are valid identifiers: `[_A-Za-z][_A-Za-z0-9]*`

The values may not contain newlines or any vertical whitespace.

The initial key/value pairs are read from the command line using repeated
`--substitute=KEY=DEFAULT-VALUE` flags.

Updated values for those keys will be read from any files provided to the
`--status-file` flag. This flag can be given multiple times and the values will
be read and updated from the files in order, meaning the last file's value will
win. New keys are never read from these files. The file format parsed is Bazel's
[status file format](https://bazel.build/docs/user-manual#workspace-status):
each line is a single entry starting with a key using only characters `[_A-Z]`,
one space character, and the rest of the line is the value. To assist with using
Bazel status files, if the key parsed from the file begins with `STABLE_`, that
prefix is removed. Any keys which are present in the substitutions provided on
the command line will have their value updated with the string read from the
file.
"""

__copyright__ = """
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

import argparse
import sys
from pathlib import Path
from string import Template


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        "--template",
        metavar="FILE",
        type=Path,
        required=True,
        help="The template source file to use.",
    )
    parser.add_argument(
        "--output",
        metavar="FILE",
        type=Path,
        required=True,
        help="The output source file to produce.",
    )
    parser.add_argument(
        "--substitution",
        metavar="KEY=DEFAULT-VALUE",
        action="append",
        help="A substitution that should be supported and its default value.",
    )
    parser.add_argument(
        "--status-file",
        metavar="FILE",
        type=Path,
        action="append",
        default=[],
        help="A file of key/value updates in Bazel's status file format.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    # Collect the supported substitutions from the command line.
    substitutions = {}
    for substitution_arg in args.substitution:
        key, value = substitution_arg.split("=", 1)
        substitutions.update({key: value})

    # Read either of the two status files provided to build up substitutions,
    # with the stable file last so its values override any duplicates.
    for status_file in args.status_file:
        if args.verbose:
            print(f"Reading status file: {status_file}", file=sys.stderr)
        for line in status_file.open():
            # Remove line endings.
            line = line.rstrip("\r\n")
            # Exactly matches our pattern
            (key, value) = line.split(" ", 1)
            key = key.removeprefix("STABLE_")
            if key in substitutions:
                if args.verbose:
                    print(f"Parsed: '{key}': '{value}'", file=sys.stderr)
                substitutions.update({key: value})

    if args.verbose:
        print(f"Reading template file: {args.template}", file=sys.stderr)
    with open(args.template) as template_file:
        template = template_file.read()

    result = Template(template).substitute(substitutions)

    if args.verbose:
        print(f"Writing output file: {args.output}", file=sys.stderr)
    with open(args.output, mode="w") as output_file:
        output_file.write(result)


if __name__ == "__main__":
    main()
