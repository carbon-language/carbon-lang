#!/usr/bin/python
#
# lit-lint
#
# Check that the RUN commands in lit tests can be executed with an emulator.
#

import argparse
import re
import sys

parser = argparse.ArgumentParser(description='lint lit tests')
parser.add_argument('filenames', nargs='+')
parser.add_argument('--filter')  # ignored
args = parser.parse_args()

runRegex = re.compile(r'(?<!-o)(?<!%run) %t\s')
errorMsg = "litlint: {}:{}: error: missing %run before %t.\n\t{}"

def LintFile(p):
    with open(p, 'r') as f:
        for i, s in enumerate(f.readlines()):
            if runRegex.search(s):
               sys.stderr.write(errorMsg.format(p, i, s))

for p in args.filenames:
    LintFile(p)
