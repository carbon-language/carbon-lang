#!/usr/bin/env python3

"""Tests for check_proposal_names.py."""

__copyright__ = """
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

import os
import pathlib
import tempfile
import unittest


from proposals.scripts import check_proposal_names


class TestCheckProposalNames(unittest.TestCase):
    def test_rename_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = pathlib.Path(tmpdir)
            proposals_dir = tmp_path / "proposals"
            proposals_dir.mkdir()

            proposal_path = proposals_dir / "p1.md"
            with open(proposal_path, "w") as f:
                f.write("# My Proposal\n")

            check_proposal_names._rename_proposals(proposals_dir)

            self.assertTrue((proposals_dir / "p000001-my-proposal.md").exists())
            self.assertFalse(proposal_path.exists())

    def test_rename_file_and_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = pathlib.Path(tmpdir)
            proposals_dir = tmp_path / "proposals"
            proposals_dir.mkdir()

            (proposals_dir / "p1").mkdir()
            proposal_path = proposals_dir / "p1.md"
            with open(proposal_path, "w") as f:
                f.write("# My Proposal\n")

            check_proposal_names._rename_proposals(proposals_dir)

            self.assertTrue((proposals_dir / "p000001-my-proposal.md").exists())
            self.assertTrue((proposals_dir / "p000001-my-proposal").exists())
            self.assertFalse(proposal_path.exists())
            self.assertFalse((proposals_dir / "p1").exists())

    def test_no_title(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = pathlib.Path(tmpdir)
            proposals_dir = tmp_path / "proposals"
            proposals_dir.mkdir()

            proposal_path = proposals_dir / "p1.md"
            with open(proposal_path, "w") as f:
                f.write("No title here\n")

            check_proposal_names._rename_proposals(proposals_dir)

            self.assertTrue((proposals_dir / "p000001-untitled.md").exists())
            self.assertFalse(proposal_path.exists())

    def test_update_references(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = pathlib.Path(tmpdir)
            proposals_dir = tmp_path / "proposals"
            proposals_dir.mkdir()

            proposal_path = proposals_dir / "p1.md"
            with open(proposal_path, "w") as f:
                f.write("# My Proposal\n")

            # Create another markdown file referencing p1.md
            ref_path = tmp_path / "README.md"
            with open(ref_path, "w") as f:
                f.write("See [proposal](proposals/p1.md)\n")

            # Run
            check_proposal_names._rename_proposals(proposals_dir)

            # Verify
            with open(ref_path, "r") as f:
                content = f.read()
            self.assertEqual(
                content, "See [proposal](proposals/p000001-my-proposal.md)\n"
            )

    def test_update_directory_references(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = pathlib.Path(tmpdir)
            proposals_dir = tmp_path / "proposals"
            proposals_dir.mkdir()

            (proposals_dir / "p1").mkdir()
            proposal_path = proposals_dir / "p1.md"
            with open(proposal_path, "w") as f:
                f.write("# My Proposal\n")

            # Create another markdown file referencing p1 directory
            ref_path = tmp_path / "README.md"
            with open(ref_path, "w") as f:
                f.write(
                    "See [dir](proposals/p1) or [file](proposals/p1.md) and "
                    "bare p1 in text.\n"
                )

            # Run
            check_proposal_names._rename_proposals(proposals_dir)

            # Verify
            with open(ref_path, "r") as f:
                content = f.read()
            self.assertEqual(
                content,
                "See [dir](proposals/p000001-my-proposal) or "
                "[file](proposals/p000001-my-proposal.md) and "
                "bare p1 in text.\n",
            )

    def test_relative_path_rename(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = pathlib.Path(tmpdir)
            old_cwd = os.getcwd()
            os.chdir(tmp_path)
            try:
                os.makedirs("proposals")
                proposal_path = pathlib.Path("proposals/p1.md")
                with open(proposal_path, "w") as f:
                    f.write("# My Proposal\n")

                check_proposal_names._rename_proposals(
                    pathlib.Path("proposals")
                )

                self.assertTrue(
                    os.path.exists("proposals/p000001-my-proposal.md")
                )
            finally:
                os.chdir(old_cwd)


if __name__ == "__main__":
    unittest.main()
