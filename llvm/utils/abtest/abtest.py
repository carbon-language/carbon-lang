#!/usr/bin/env python
#
# Given a previous good compile narrow down miscompiles.
# Expectes two directories named "before" and "after" each containing a set of
# assembly files where the "after" version is assumed to be broken.
# Also assumes the presence of a executable or script "link_test" which when
# called with a set of assembly files will link them together and test if the
# resulting executable is "good".
#
# Example usage:
#    1. Create a link_test script, make it executable. Simple Example:
#          clang "$@" -o /tmp/test && /tmp/test || echo "PROBLEM"
#    2. Run the script to figure out which files are miscompiled:
#       > ./abtest.py 
#       somefile.s: ok
#       someotherfile.s: skipped: same content
#       anotherfile.s: failed: './link_test' exitcode != 0
#       ...
#    3. If you want to replace+test with the functions inside a single file
#       you first have to mark function begins and ends in the .s files
#       this directory comes with some: mark_XXX.py example scripts,
#       unfortunately you usually have to adapt them to each environment.
#       > for i in before/*.s after/*.s; do mark_xxx.py $i; done
#    4. Run the tests on a single file
#       > ./abtest.py
#       funcname1 [0/XX]: ok
#       funcname2 [1/XX]: ok
#       funcname3 [2/XX]: skipped: same content
#       funcname4 [3/XX]: failed: './link_test' exitcode != 0
#       ...
import sys
import os
from os import system, mkdir, walk, makedirs, errno, getenv
from os.path import dirname, isdir
from shutil import rmtree, copyfile
from fnmatch import filter
from itertools import chain
from subprocess import call
from sys import stderr
import argparse
import filecmp

LINKTEST="./link_test"
ESCAPE="\033[%sm"
BOLD=ESCAPE % "1"
RED=ESCAPE % "31"
NORMAL=ESCAPE % "0"
FAILED=RED+"failed"+NORMAL

def mkdirtree(path):
    try:
        makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and isdir(path):
            pass
        else:
            raise

def find(dir, file_filter=None):
    files = [walkdir[0]+"/"+file for walkdir in walk(dir) for file in walkdir[2]]
    if file_filter != None:
        files = filter(files, file_filter)
    return files

def error(message):
    stderr.write("Error: %s\n" % (message,))

def warn(message):
    stderr.write("Warning: %s\n" % (message,))

def notice(message):
    stderr.write("%s\n" % message)

def extract_functions(file):
    functions = []
    in_function = None
    for line in open(file):
        if line.startswith("# -- Begin  "):
            if in_function != None:
                warn("Missing end of function %s" % (in_function,))
            funcname = line[12:-1]
            in_function = funcname
            text = line
        elif line.startswith("# -- End  "):
            function_name = line[10:-1]
            if in_function != function_name:
                warn("End %s does not match begin %s" % (function_name, in_function))
            else:
                text += line
                functions.append( (in_function, text) )
            in_function = None
        elif in_function != None:
            text += line
    return functions

def replace_function(file, function, replacement, dest):
    out = open(dest, "w")
    skip = False
    found = False
    in_function = None
    for line in open(file):
        if line.startswith("# -- Begin  "):
            if in_function != None:
                warn("Missing end of function %s" % (in_function,))
            funcname = line[12:-1]
            in_function = funcname
            if in_function == function:
                out.write(replacement)
                skip = True
        elif line.startswith("# -- End  "):
            function_name = line[10:-1]
            if in_function != function_name:
                warn("End %s does not match begin %s" % (function_name, in_function))
            in_function = None
            if skip:
                skip = False
                continue
        if not skip:
            out.write(line)

def announce_test(name):
    stderr.write("%s%s%s: " % (BOLD, name, NORMAL))
    stderr.flush()

def announce_result(result, info):
    stderr.write(result)
    if info != "":
        stderr.write(": %s" % info)
    stderr.write("\n")
    stderr.flush()

def testrun(files):
    linkline="%s %s" % (LINKTEST, " ".join(files),)
    res = call(linkline, shell=True)
    if res != 0:
        announce_result(FAILED, "'%s' exitcode != 0" % LINKTEST)
    else:
        announce_result("ok", "")

def check_files():
    """Check files mode"""
    for i in range(0, len(NO_PREFIX)):
        f = NO_PREFIX[i]
        b=baddir+"/"+f
        if b not in BAD_FILES:
            warn("There is no corresponding file to '%s' in %s" \
                 % (gooddir+"/"+f, baddir))
            continue

        announce_test(f + " [%s/%s]" % (i+1, len(NO_PREFIX)))

        # combine files (everything from good except f)
        testfiles=[]
        skip=False
        for c in NO_PREFIX:
            badfile = baddir+"/"+c
            goodfile = gooddir+"/"+c
            if c == f:
                testfiles.append(badfile)
                if filecmp.cmp(goodfile, badfile):
                    announce_result("skipped", "same content")
                    skip = True
                    break
            else:
                testfiles.append(goodfile)
        if skip:
            continue
        testrun(testfiles)

def check_functions_in_file(base, goodfile, badfile):
    functions = extract_functions(goodfile)
    if len(functions) == 0:
        warn("Couldn't find any function in %s, missing annotations?" % (goodfile,))
        return
    badfunctions = dict(extract_functions(badfile))
    if len(functions) == 0:
        warn("Couldn't find any function in %s, missing annotations?" % (badfile,))
        return

    COMBINED="/tmp/combined.s"
    i = 0
    for (func,func_text) in functions:
        announce_test(func + " [%s/%s]" % (i+1, len(functions)))
        i+=1
        if func not in badfunctions:
            warn("Function '%s' missing from bad file" % func)
            continue
        if badfunctions[func] == func_text:
            announce_result("skipped", "same content")
            continue
        replace_function(goodfile, func, badfunctions[func], COMBINED)
        testfiles=[]
        for c in NO_PREFIX:
            if c == base:
                testfiles.append(COMBINED)
                continue
            testfiles.append(gooddir + "/" + c)

        testrun(testfiles)

parser = argparse.ArgumentParser()
parser.add_argument('--a', dest='dir_a', default='before')
parser.add_argument('--b', dest='dir_b', default='after')
parser.add_argument('file', metavar='file', nargs='?')
config = parser.parse_args()

# Check if environment is sane
if not os.access(LINKTEST, os.X_OK):
    error("Expect '%s' to be present and executable" % (LINKTEST,))
    exit(1)

gooddir=config.dir_a
baddir=config.dir_b

BAD_FILES=find(baddir, "*")
GOOD_FILES=find(gooddir, "*")
NO_PREFIX=sorted([x[len(gooddir)+1:] for x in GOOD_FILES])

if config.file is not None:
    goodfile = gooddir+"/"+config.file
    badfile = baddir+"/"+config.file
    check_functions_in_file(config.file, goodfile, badfile)
else:
    check_files()
