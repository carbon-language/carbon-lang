#!/usr/bin/python

import os
import sys

DBG_OUTPUT_FILE="Output/" + sys.argv[1] + ".dbg.out"
OPT_DBG_OUTPUT_FILE="Output/" + sys.argv[1] + ".dbg.opt.out"
LOG_FILE="Output/" + sys.argv[1] + ".log"
NATIVE_DBG_OUTPUT_FILE="Output/" + sys.argv[1] + ".native.dbg.out"
NATIVE_OPT_DBG_OUTPUT_FILE="Output/" + sys.argv[1] + ".native.dbg.opt.out"
NATIVE_LOG_FILE="Output/" + sys.argv[1] + ".native.log"
REPORT_FILE="Output/" + sys.argv[1] + ".dbg.report.html"

class BreakPoint:
    def __init__(self, bp_name):
        self.name = bp_name
        self.values = {}
        self.missing_args = []
        self.matching_args = []
        self.notmatching_args = []
        self.missing_bp = False

    def setMissing(self):
        self.missing_bp = True

    def getArgCount(self):
        return len(self.values)

    def getMissingArgCount(self):
        if self.missing_bp == True:
            return len(self.values)
        return len(self.missing_args)

    def getMatchingArgCount(self):
        if self.missing_bp == True:
            return 0
        return len(self.matching_args)

    def getNotMatchingArgCount(self):
        if self.missing_bp == True:
            return 0
        return len(self.notmatching_args)

    def recordArgument(self, arg_name, value):
        self.values[arg_name] = value
        
    def __repr__(self):
        print self.name
        items = self.values.items()
        for i in range(len(items)):
            print items[i][0]," = ",items[i][1]
        return ''

    def compare_args(self, other, file):
        myitems = self.values.items()
        otheritems = other.values.items()
        match = False
        for i in range(len(myitems)):
            if i >= len(otheritems):
                match = True
                self.missing_args.append(myitems[i][0])
            elif cmp(myitems[i][1], otheritems[i][1]):
                match = True
                self.notmatching_args.append(myitems[i][0])
            else:
                self.matching_args.append(myitems[i][0])

        self.print_list(self.matching_args, " Matching arguments ", file)
        self.print_list(self.notmatching_args, " Not Matching arguments ", file)
        self.print_list(self.missing_args, " Missing arguments ", file)
        return match

    def print_list(self, items, txt, pfile):
        if len(items) == 0:
            return
        pfile.write(self.name)
        pfile.write(txt)
        for e in items:
            pfile.write(e)
            pfile.write(' ')
        pfile.write('\n')

def read_input(filename, dict):
    f = open(filename, "r")
    lines = f.readlines()
    for l in range(len(lines)):
        c = lines[l].split()
        if c[0] == "#Breakpoint":
            bp = dict.get(c[2])
            if bp is None:
                bp = BreakPoint(c[1])
            dict[c[2]] = bp
        if c[0] == "#Argument":
            bp = dict.get(c[2])
            if bp is None:
                bp = BreakPoint(c[1])
            dict[c[2]] = bp
            bp.recordArgument(c[3], c[4])
    return

f1_breakpoints = {}
read_input(DBG_OUTPUT_FILE, f1_breakpoints)
f1_items = f1_breakpoints.items()

f2_breakpoints = {}
read_input(OPT_DBG_OUTPUT_FILE, f2_breakpoints)
f2_items = f2_breakpoints.items()
    
f = open(LOG_FILE, "w")
f.write("Log output\n")
for f2bp in range(len(f2_items)):
    id = f2_items[f2bp][0]
    bp = f2_items[f2bp][1]
    bp1 = f1_breakpoints.get(id)
    if bp1 is None:
        bp.setMissing()
    else:
        bp1.compare_args(bp,f)
f.close()

nf1_breakpoints = {}
read_input(NATIVE_DBG_OUTPUT_FILE, nf1_breakpoints)
nf1_items = nf1_breakpoints.items()

nf2_breakpoints = {}
read_input(NATIVE_OPT_DBG_OUTPUT_FILE, nf2_breakpoints)
nf2_items = nf2_breakpoints.items()
    
nfl = open(NATIVE_LOG_FILE, "w")
for nf2bp in range(len(nf2_items)):
    id = nf2_items[nf2bp][0]
    bp = nf2_items[nf2bp][1]
    bp1 = nf1_breakpoints.get(id)
    if bp1 is None:
        bp.setMissing()
    else:
        bp1.compare_args(bp,nfl)
nfl.close()

f1_arg_count = 0
f1_matching_arg_count = 0
f1_notmatching_arg_count = 0
f1_missing_arg_count = 0
for idx in range(len(f1_items)):
    bp = f1_items[idx][1]
    f1_arg_count = f1_arg_count + bp.getArgCount()
    f1_matching_arg_count = f1_matching_arg_count + bp.getMatchingArgCount()
    f1_notmatching_arg_count = f1_notmatching_arg_count + bp.getNotMatchingArgCount()
    f1_missing_arg_count = f1_missing_arg_count + bp.getMissingArgCount()

nf1_arg_count = 0
nf1_matching_arg_count = 0
nf1_notmatching_arg_count = 0
nf1_missing_arg_count = 0
for idx in range(len(nf1_items)):
    bp = nf1_items[idx][1]
    nf1_arg_count = nf1_arg_count + bp.getArgCount()
    nf1_matching_arg_count = nf1_matching_arg_count + bp.getMatchingArgCount()
    nf1_notmatching_arg_count = nf1_notmatching_arg_count + bp.getNotMatchingArgCount()
    nf1_missing_arg_count = nf1_missing_arg_count + bp.getMissingArgCount()

rf = open(REPORT_FILE, "w")
rf.write("<tr><td>")
rf.write(str(sys.argv[1]))
rf.write("</td><td>|</td><td>")
rf.write(str(nf1_arg_count))
rf.write("</td><td><b>")
rf.write(str(nf1_matching_arg_count))
rf.write("</b></td><td>")
rf.write(str(nf1_notmatching_arg_count))
rf.write("</td><td>")
rf.write(str(nf1_missing_arg_count))
rf.write("</td><td>|</td><td>")
rf.write(str(f1_arg_count))
rf.write("</td><td><b>")
rf.write(str(f1_matching_arg_count))
rf.write("</b></td><td>")
rf.write(str(f1_notmatching_arg_count))
rf.write("</td><td>")
rf.write(str(f1_missing_arg_count))
rf.write("\n")
rf.close()
