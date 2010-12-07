#!/usr/bin/python

import os
import sys

class BreakPoint:
    def __init__(self, bp_name):
        self.name = bp_name
        self.values = {}

    def recordArgument(self, arg_name, value):
        self.values[arg_name] = value
        
    def __repr__(self):
        print self.name
        items = self.values.items()
        for i in range(len(items)):
            print items[i][0]," = ",items[i][1]
        return ''

    def __cmp__(self, other):
        return cmp(self.values, other.values)

def read_input(filename, dict):
    f = open(filename, "r")
    lines = f.readlines()
    for l in range(len(lines)):
        c = lines[l].split()
        if c[0] == "#Argument":
            bp = dict.get(c[2])
            if bp is None:
                bp = BreakPoint(c[1])
            dict[c[2]] = bp
            bp.recordArgument(c[3], c[4])

    f.close()
    return

f1_breakpoints = {}
read_input(sys.argv[1], f1_breakpoints)
f1_items = f1_breakpoints.items()

f2_breakpoints = {}
read_input(sys.argv[2], f2_breakpoints)
f2_items = f2_breakpoints.items()
    
mismatch = 0
for f2bp in range(len(f2_items)):
    id = f2_items[f2bp][0]
    bp = f2_items[f2bp][1]
    bp1 = f1_breakpoints.get(id)
    if bp1 is None:
        print "bp is missing"
    else:
        if bp1 != bp:
            mismatch = mismatch + 1

l2 = len(f2_items)
print "=========="
if l2 != 0:
    print sys.argv[3]," success rate is", (l2-mismatch)*100/l2,"%"
else:
    print sys.argv[3]," success rate is 100%"
print "=========="
