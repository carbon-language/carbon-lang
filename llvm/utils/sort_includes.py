#!/usr/bin/env python

"""Script to sort the top-most block of #include lines.

Assumes the LLVM coding conventions.

Currently, this script only bothers sorting the llvm/... headers. Patches
welcome for more functionality, and sorting other header groups.
"""

import argparse
import os

def sort_includes(f):
  """Sort the #include lines of a specific file."""

  # Skip files which are under INPUTS trees or test trees.
  if 'INPUTS/' in f.name or 'test/' in f.name:
    return

  lines = f.readlines()
  look_for_api_header = os.path.splitext(f.name)[1] == '.cpp'
  found_headers = False
  headers_begin = 0
  headers_end = 0
  api_headers = []
  local_headers = []
  project_headers = []
  system_headers = []
  for (i, l) in enumerate(lines):
    if l.strip() == '':
      continue
    if l.startswith('#include'):
      if not found_headers:
        headers_begin = i
        found_headers = True
      headers_end = i
      header = l[len('#include'):].lstrip()
      if look_for_api_header and header.startswith('"'):
        api_headers.append(header)
        look_for_api_header = False
        continue
      if header.startswith('<') or header.startswith('"gtest/'):
        system_headers.append(header)
        continue
      if (header.startswith('"llvm/') or header.startswith('"llvm-c/') or
          header.startswith('"clang/') or header.startswith('"clang-c/')):
        project_headers.append(header)
        continue
      local_headers.append(header)
      continue

    # Only allow comments and #defines prior to any includes. If either are
    # mixed with includes, the order might be sensitive.
    if found_headers:
      break
    if l.startswith('//') or l.startswith('#define') or l.startswith('#ifndef'):
      continue
    break
  if not found_headers:
    return

  local_headers.sort()
  project_headers.sort()
  system_headers.sort()
  headers = api_headers + local_headers + project_headers + system_headers
  header_lines = ['#include ' + h for h in headers]
  lines = lines[:headers_begin] + header_lines + lines[headers_end + 1:]

  f.seek(0)
  f.truncate()
  f.writelines(lines)

def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('files', nargs='+', type=argparse.FileType('r+'),
                      help='the source files to sort includes within')
  args = parser.parse_args()
  for f in args.files:
    sort_includes(f)

if __name__ == '__main__':
  main()
