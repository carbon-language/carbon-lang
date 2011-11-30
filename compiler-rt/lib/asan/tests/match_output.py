#!/usr/bin/python

import re
import sys

def matchFile(f, f_re):
  for line_re in f_re:
    line_re = line_re.rstrip()
    if not line_re:
      continue
    if line_re[0] == '#':
      continue
    match = False
    for line in f:
      line = line.rstrip()
      # print line
      if re.search(line_re, line):
        match = True
        #print 'match: %s =~ %s' % (line, line_re)
        break
    if not match:
      print 'no match for: %s' % (line_re)
      return False
  return True

if len(sys.argv) != 2:
  print >>sys.stderr, 'Usage: %s <template file>'
  sys.exit(1)

f = sys.stdin
f_re = open(sys.argv[1])

if not matchFile(f, f_re):
  print >>sys.stderr, 'File does not match the template'
  sys.exit(1)
