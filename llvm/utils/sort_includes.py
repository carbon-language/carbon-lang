#!/usr/bin/env python

"""Script to sort the top-most block of #include lines.

Assumes the LLVM coding conventions.

Currently, this script only bothers sorting the llvm/... headers. Patches
welcome for more functionality, and sorting other header groups.
"""

import argparse
import os
import re
import sys
import tempfile

def sort_includes(f):
  lines = f.readlines()
  look_for_api_header = f.name[-4:] == '.cpp'
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
      if headers_begin == 0:
        headers_begin = i
      headers_end = i
      header = l[len('#include'):].lstrip()
      if look_for_api_header and header.startswith('"'):
        api_headers.append(header)
        look_for_api_header = False
        continue
      if header.startswith('<'):
        system_headers.append(header)
        continue
      if header.startswith('"llvm/') or header.startswith('"clang/'):
        project_headers.append(header)
        continue
      local_headers.append(header)
      continue

    # Only allow comments and #defines prior to any includes. If either are
    # mixed with includes, the order might be sensitive.
    if headers_begin != 0:
      break
    if l.startswith('//') or l.startswith('#define') or l.startswith('#ifndef'):
      continue
    break
  if headers_begin == 0:
    return

  local_headers.sort()
  project_headers.sort()
  system_headers.sort()
  headers = api_headers + local_headers + project_headers + system_headers
  header_lines = ['#include ' + h for h in headers]
  lines = lines[:headers_begin] + header_lines + lines[headers_end + 1:]

  #for l in lines[headers_begin:headers_end]:
  #  print l.rstrip()
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
