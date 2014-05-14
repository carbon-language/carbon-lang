#!/usr/bin/python
#
# lit-lint
#
# Check that the RUN commands in lit tests can be executed with an emulator.
#

import optparse
import re
import sys

parser = optparse.OptionParser()
parser.add_option('--filter')  # ignored
(options, filenames) = parser.parse_args()

runRegex = re.compile(r'(?<!-o)(?<!%run) %t\s')
errorMsg = "litlint: {}:{}: error: missing %run before %t.\n\t{}"

def LintFile(p):
    with open(p, 'r') as f:
        for i, s in enumerate(f.readlines()):
            if runRegex.search(s):
               sys.stderr.write(errorMsg.format(p, i, s))

for p in filenames:
    LintFile(p)
