#!/usr/bin/env python

import sys

# Currently any print-out from the custom tool is interpreted as a crash
# (i.e. test is still interesting)

print "Error: " + ' '.join(sys.argv[1:])

sys.exit(1)
