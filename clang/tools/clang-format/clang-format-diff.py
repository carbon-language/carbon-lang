#!/usr/bin/python
#
#===- clang-format-diff.py - ClangFormat Diff Reformatter ----*- python -*--===#
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
#===------------------------------------------------------------------------===#

r"""
ClangFormat Diff Reformatter
============================

This script reads input from a unified diff and reformats all the changed
lines. This is useful to reformat all the lines touched by a specific patch.
Example usage for git users:

  git diff -U0 HEAD^ | clang-format-diff.py -p1

"""

import argparse
import re
import subprocess
import sys


# Change this to the full path if clang-format is not on the path.
binary = 'clang-format'


def main():
  parser = argparse.ArgumentParser(description=
                                   'Reformat changed lines in diff.')
  parser.add_argument('-p', default=0,
                      help='strip the smallest prefix containing P slashes')
  parser.add_argument(
      '-style',
      help=
      'formatting style to apply (LLVM, Google, Chromium, Mozilla, WebKit)')
  args = parser.parse_args()

  # Extract changed lines for each file.
  filename = None
  lines_by_file = {}
  for line in sys.stdin:
    match = re.search('^\+\+\+\ (.*?/){%s}(\S*)' % args.p, line)
    if match:
      filename = match.group(2)
    if filename == None:
      continue

    # FIXME: Add other types containing C++/ObjC code.
    if not (filename.endswith(".cpp") or filename.endswith(".cc") or
            filename.endswith(".h")):
      continue

    match = re.search('^@@.*\+(\d+)(,(\d+))?', line)
    if match:
      start_line = int(match.group(1))
      line_count = 1
      if match.group(3):
        line_count = int(match.group(3))
      if line_count == 0:
        continue
      end_line = start_line + line_count - 1;
      lines_by_file.setdefault(filename, []).extend(
          ['-lines', str(start_line) + ':' + str(end_line)])

  # Reformat files containing changes in place.
  for filename, lines in lines_by_file.iteritems():
    command = [binary, '-i', filename]
    command.extend(lines)
    if args.style:
      command.extend(['-style', args.style])
    p = subprocess.Popen(command, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE,
                         stdin=subprocess.PIPE)
    stdout, stderr = p.communicate()
    if stderr:
      print stderr
    if p.returncode != 0:
      sys.exit(p.returncode);


if __name__ == '__main__':
  main()
