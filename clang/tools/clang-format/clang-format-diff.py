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


def getOffsetLength(filename, line_number, line_count):
  """
  Calculates the field offset and length based on line number and count.
  """
  offset = 0
  length = 0
  with open(filename, 'r') as f:
    for line in f:
      if line_number > 1:
        offset += len(line)
        line_number -= 1
      elif line_count > 0:
        length += len(line)
        line_count -= 1
      else:
        break
  return offset, length


def formatRange(r, style):
  """
  Formats range 'r' according to style 'style'.
  """
  filename, line_number, line_count = r
  # FIXME: Add other types containing C++/ObjC code.
  if not (filename.endswith(".cpp") or filename.endswith(".cc") or
          filename.endswith(".h")):
    return

  offset, length = getOffsetLength(filename, line_number, line_count)
  with open(filename, 'r') as f:
    text = f.read()
  command = [binary, '-offset', str(offset), '-length', str(length)]
  if style:
    command.extend(['-style', style])
  p = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       stdin=subprocess.PIPE)
  stdout, stderr = p.communicate(input=text)
  if stderr:
    print stderr
    return
  if not stdout:
    print 'Segfault occurred while formatting', filename
    print 'Please report a bug on llvm.org/bugs.'
    return
  with open(filename, 'w') as f:
    f.write(stdout)


def main():
  parser = argparse.ArgumentParser(description=
                                   'Reformat changed lines in diff')
  parser.add_argument('-p', default=0,
                      help='strip the smallest prefix containing P slashes')
  parser.add_argument('-style',
                      help='formatting style to apply (LLVM, Google, Chromium)')
  args = parser.parse_args()

  filename = None
  ranges = []

  for line in sys.stdin:
    match = re.search('^\+\+\+\ (.*?/){%s}(\S*)' % args.p, line)
    if match:
      filename = match.group(2)
    if filename == None:
      continue

    match = re.search('^@@.*\+(\d+)(,(\d+))?', line)
    if match:
      line_count = 1
      if match.group(3):
        line_count = int(match.group(3))
      ranges.append((filename, int(match.group(1)), line_count))

  # Reverse the ranges so that the reformatting does not influence file offsets.
  for r in reversed(ranges):
    # Do the actual formatting.
    formatRange(r, args.style)


if __name__ == '__main__':
  main()
