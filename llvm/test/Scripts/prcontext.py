#
# Usage:
# prcontext <pattern> <# lines of context>
#

import sys

#
# Get the arguments
#
pattern=sys.argv[1]
num=int(sys.argv[2])

#
# Get all of the lines in the file.
#
lines=sys.stdin.readlines()

index=0
for line in lines:
  if ((line.find(pattern)) != -1):
    if (index-num < 0):
      bottom=0
    else:
      bottom=index-num
    for output in lines[bottom:index+num+1]:
      print output[:-1]
  index=index+1

