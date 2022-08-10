#!/usr/bin/env python3

"""Generate Explorer test cases from Markdown code snippets"""

__copyright__ = """
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

import argparse
import glob
from html.parser import HTMLParser
import os
from pathlib import Path
from markdown import Markdown, Extension, markdownFromFile
from markdown.extensions.fenced_code import FencedCodeExtension
from markdown.preprocessors import Preprocessor
from markdown.postprocessors import Postprocessor
import re
from typing import Any, Dict, List, Tuple

TEST_HEADER = """
// RUN: %{explorer} %s 2>&1 | \\
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes=false %s
// RUN: %{explorer} --parser_debug --trace_file=- %s 2>&1 | \\
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes %s
// AUTOUPDATE: %{explorer} %s
"""
NAMED_SNIPPETS = {
    "m": ["fn Main() -> i32 { return 0; }"],
    "mo": ["fn Main() -> i32 {"],
    "rc": ["return 0; }"],
    "r": ["return 0;"],
    "c": ["}"],
}
RE_TEST = re.compile(r"\s*test\s+")
RE_TEST_COMMENT = re.compile(r"\s*<!--\s*test\s+")
RE_TEST_COMMAND = re.compile(r"""\s*(?P<command>
 (?P<out_buf>[_])
|`(?P<out_code>[^`]+)`
|(?P<out_name>[\w_][\w\d_]*)
|[-](?P<del_line>[\d]+)([+](?P<del_lines>[\d]+))?
|[+](?P<ins_line>[\d]+)(`(?P<ins_code>[^`]+)`|(?P<ins_name>[\w_][\w\d_]*))
|[.]`(?P<dot_code>[^`]+)`
|[.](?P<dot_name>[\w_][\w\d_]*)
|(?P<dot_none>[.])
|[=](?P<cpy_name>[\w_][\w\d_]*)
)""", re.VERBOSE)
RE_DOTS = re.compile(r"[.]{3}")

def parse_test(text : str) -> List[Dict]:
  if not (match := RE_TEST.match(text)):
    return None
  matches = []
  while match := RE_TEST_COMMAND.match(text, match.end()):
    matches.append(match.groupdict())
  return matches

class TestLinenoProcessor(Preprocessor):
    def __init__(self, linenos: List[int]) -> None:
        super().__init__()
        self.linenos = linenos

    def run(self, lines) -> List[str]:
        lineno = 0
        for line in lines:
            lineno += 1
            if RE_TEST_COMMENT.match(line):
                self.linenos.append(lineno)
        return lines

class DocsSnippetParser(HTMLParser):
    def __init__(self, test_linenos: List[int], outdir: Path) -> None:
        super().__init__()
        self.outdir = outdir
        self.comm_lineno = 0
        self.code_lineno = 0
        self.code_text = None
        self.code_lines = None
        self.snippets = NAMED_SNIPPETS
        self.test_text = None
        self.test_cmds = None
        self.test_lines = None
        self.test_linenos = test_linenos
        self.test_index = 0
        self.test_lineno = 0
    def handle_comment(self, data: str) -> None:
        if RE_TEST.match(data):
          self.test_text = data
          self.comm_lineno = self.getpos()[0]
          self.test_lineno = self.test_linenos[self.test_index]
          self.test_index += 1
    def handle_starttag(self, tag: str, attrs: List[Tuple[str, str]]) -> None:
        langs = [v for k, v in attrs if k == "class" and v.startswith("lang-")]
        if tag == "code" and (not langs or "lang-carbon" in langs):
            self.code_lineno = self.getpos()[0]
    def handle_data(self, data: str) -> None:
        #TODO: there is no error handling
        if self.code_lineno > 0 and self.comm_lineno >= self.code_lineno - 2 \
          and self.test_text and (test_cmds := parse_test(self.test_text)):
            self.code_text = data
            self.code_lines = self.code_text.splitlines()
            self.test_cmds = test_cmds
            self.test_lines = []
            for cmd in self.test_cmds:
              if cmd["out_buf"]:
                self.test_lines.extend(self.code_lines)
              elif cmd["out_code"]:
                lines = cmd["out_code"].splitlines()
                self.test_lines.extend(lines)
              elif cmd["out_name"]:
                self.test_lines.extend(self.snippets[cmd["out_name"]])
              elif cmd["del_lines"]:
                del_line = int(cmd["del_line"]) - 1
                del_count = int(cmd["del_lines"])
                del self.code_lines[del_line:del_line + del_count - 1]
              elif cmd["del_line"]:
                del_line = int(cmd["del_line"]) - 1
                del self.code_lines[del_line]
              elif cmd["ins_code"]:
                ins_line = int(cmd["ins_line"]) - 1
                lines = cmd["ins_code"].splitlines()
                self.code_lines = self.code_lines[:ins_line] + lines + self.code_lines[:ins_line]
              elif cmd["ins_name"]:
                ins_line = int(cmd["ins_line"]) - 1
                lines = self.snippets[cmd["ins_name"]]
                self.code_lines = self.code_lines[:ins_line] + lines + self.code_lines[ins_line:]
              elif cmd["dot_code"]:
                code = cmd["dot_code"]
                self.code_text = RE_DOTS.sub(code, self.code_text)
                self.code_lines = self.code_text.splitlines()
              elif cmd["dot_name"]:
                code = self.snippets[cmd["dot_name"]]
                self.code_text = RE_DOTS.sub(code, self.code_text)
                self.code_lines = self.code_text.splitlines()
              elif cmd["dot_none"]:
                self.code_text = RE_DOTS.sub("", self.code_text)
                self.code_lines = self.code_text.splitlines()
              elif cmd["cpy_name"]:
                name = cmd["cpy_name"]
                self.snippets[name] = self.code_lines

    def handle_endtag(self, tag: str) -> None:
        if self.test_lines:
            testpath = os.path.join(self.outdir, f"line_{self.test_lineno}.carbon")
            with open(testpath, "w", encoding="utf-8") as testfile:
                testfile.write(TEST_HEADER)
                testfile.write(f"// CHECK: result: 0") #TODO: make configurable
                testfile.write("\n")
                testfile.write("package ExplorerTest api;\n")
                testfile.write("\n")
                testfile.writelines(map(lambda l: l + "\n", self.test_lines))
                testfile.write("\n")
        self.code_lineno = 0
        self.code_text = None
        self.code_lines = None
        self.test_text = None
        self.test_data = None
        self.test_lines = None
        self.test_lineno = 0

class DocsSnippetProcessor(Postprocessor):
    def __init__(self, md: Markdown, test_linenos: List[int], outdir) -> None:
        self.md = md
        self.test_linenos = test_linenos
        self.outdir = outdir
    def run(self, text: str) -> str:
        parser = DocsSnippetParser(self.test_linenos, self.outdir)
        parser.feed(text)
        return text

class DocsSnippetToTest(Extension):
    def __init__(self, path, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.path = path
    def extendMarkdown(self, md: Markdown) -> None:
        test_linenos = []
        md.preprocessors.register(TestLinenoProcessor(test_linenos), "test_to_lineno", 999)
        md.postprocessors.register(DocsSnippetProcessor(md, test_linenos, self.path), "md_to_lit", 0)
        md.registerExtension(self)

def main() -> None:
    arg_parser = argparse.ArgumentParser(description=__doc__)
    arg_parser.add_argument(
        "--debug", help="Test generator commands to debug."
    )
    arg_parser.add_argument(
        "--input", help="Markdown file to parse."
    )
    arg_parser.add_argument(
        "--output", help="Output directory. NOTE: contents will be cleared!"
    )
    args = arg_parser.parse_args()
    if args.debug:
      if matches := parse_test(args.debug):
        for match in matches:
          print({k: v for (k, v) in match.items() if v})
      exit(0)

    outpath = Path(args.output).resolve()
    os.makedirs(outpath, exist_ok=True)
    outfiles = glob.glob(f"{args.output}/*.carbon")
    for f in outfiles:
        os.remove(f)

    md_extensions = [FencedCodeExtension(lang_prefix="lang-"), DocsSnippetToTest(outpath)]
    markdownFromFile(input=args.input, extensions=md_extensions, output=os.devnull)

if __name__ == "__main__":
    main()
