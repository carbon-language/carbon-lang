"""Provides a list of proposal files."""

__copyright__ = """
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

import os
import re
import sys


def get_title(parent_dir, entry):
    """Returns the title from the requested file."""
    path = os.path.join(parent_dir, entry)
    with open(path) as md:
        titles = [t for t in md.readlines() if t.startswith("# ")]
        if not titles:
            sys.exit("%r is missing a title." % path)
        return titles[0][2:-1]


def get_path():
    return os.path.realpath(os.path.join(os.path.dirname(__file__), ".."))


def get_list(proposals_path):
    proposals = []
    proposals_list = os.listdir(proposals_path)
    proposals_list.sort()
    for f in proposals_list:
        match = re.match(r"^p([0-9]{4})\.md$", f)
        if not match:
            continue
        number = match[1]
        title = get_title(proposals_path, f)
        proposals.append(("%s - %s" % (number, title), f))
        decision_file = "p%s_decision.md" % number
        if decision_file in proposals_list:
            proposals.append(("%s - Decision" % number, decision_file))
    return proposals
