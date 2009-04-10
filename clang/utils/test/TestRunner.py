#!/usr/bin/python
#
#  TestRunner.py - This script is used to run arbitrary unit tests.  Unit
#  tests must contain the command used to run them in the input file, starting
#  immediately after a "RUN:" string.
#
#  This runner recognizes and replaces the following strings in the command:
#
#     %s - Replaced with the input name of the program, or the program to
#          execute, as appropriate.
#     %S - Replaced with the directory where the input resides.
#     %llvmgcc - llvm-gcc command
#     %llvmgxx - llvm-g++ command
#     %prcontext - prcontext.tcl script
#     %t - temporary file name (derived from testcase name)
#

import os
import sys
import subprocess
import errno
import re

class TestStatus:
    Pass = 0 
    XFail = 1
    Fail = 2
    XPass = 3
    NoRunLine = 4 
    Invalid = 5

    kNames = ['Pass','XFail','Fail','XPass','NoRunLine','Invalid']
    @staticmethod
    def getName(code): return TestStatus.kNames[code]

def mkdir_p(path):
    if not path:
        pass
    elif os.path.exists(path):
        pass
    else:
        parent = os.path.dirname(path) 
        if parent != path:
            mkdir_p(parent)
        try:
            os.mkdir(path)
        except OSError,e:
            if e.errno != errno.EEXIST:
                raise

def remove(path):
    try:
        os.remove(path)
    except OSError:
        pass

def cat(path, output):
    f = open(path)
    output.writelines(f)
    f.close()

def runOneTest(FILENAME, SUBST, OUTPUT, TESTNAME, CLANG, 
               useValgrind=False,
               useDGCompat=False,
               useScript=None, 
               output=sys.stdout):
    if useValgrind:
        VG_OUTPUT = '%s.vg'%(OUTPUT,)
        if os.path.exists:
            remove(VG_OUTPUT)
        CLANG = 'valgrind --leak-check=full --quiet --log-file=%s %s'%(VG_OUTPUT, CLANG)

    # Create the output directory if it does not already exist.
    mkdir_p(os.path.dirname(OUTPUT))

    # FIXME
    #ulimit -t 40

    # FIXME: Load script once
    # FIXME: Support "short" script syntax

    if useScript:
        scriptFile = useScript
    else:
        # See if we have a per-dir test script.
        dirScriptFile = os.path.join(os.path.dirname(FILENAME), 'test.script')
        if os.path.exists(dirScriptFile):
            scriptFile = dirScriptFile
        else:
            scriptFile = FILENAME
            
    # Verify the script contains a run line.
    for ln in open(scriptFile):
        if 'RUN:' in ln:
            break
    else:
        print >>output, "******************** TEST '%s' HAS NO RUN LINE! ********************"%(TESTNAME,)
        output.flush()
        return TestStatus.NoRunLine

    OUTPUT = os.path.abspath(OUTPUT)
    FILENAME = os.path.abspath(FILENAME)
    SCRIPT = OUTPUT + '.script'
    TEMPOUTPUT = OUTPUT + '.tmp'

    substitutions = [('%s',SUBST),
                     ('%S',os.path.dirname(SUBST)),
                     ('%llvmgcc','llvm-gcc -emit-llvm -w'),
                     ('%llvmgxx','llvm-g++ -emit-llvm -w'),
                     ('%prcontext','prcontext.tcl'),
                     ('%t',TEMPOUTPUT),
                     ('clang',CLANG)]
    scriptLines = []
    xfailLines = []
    for ln in open(scriptFile):
        if 'RUN:' in ln:
            # Isolate run parameters
            index = ln.index('RUN:')
            ln = ln[index+4:]

            # Apply substitutions
            for a,b in substitutions:
                ln = ln.replace(a,b)

            if useDGCompat:
                ln = re.sub(r'\{(.*)\}', r'"\1"', ln)
            scriptLines.append(ln)
        elif 'XFAIL' in ln:
            xfailLines.append(ln)
    
    if xfailLines:
        print >>output, "XFAILED '%s':"%(TESTNAME,)
        output.writelines(xfailLines)

    # Write script file
    f = open(SCRIPT,'w')
    f.write(''.join(scriptLines))
    f.close()

    outputFile = open(OUTPUT,'w')
    p = None
    try:
        p = subprocess.Popen(["/bin/sh",SCRIPT],
                             cwd=os.path.dirname(FILENAME),
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)
        out,err = p.communicate()
        outputFile.write(out)
        outputFile.write(err)
        SCRIPT_STATUS = p.wait()
    except KeyboardInterrupt:
        if p is not None:
            os.kill(p.pid)
        raise
    outputFile.close()

    if xfailLines:
        SCRIPT_STATUS = not SCRIPT_STATUS

    if useValgrind:
        VG_STATUS = len(list(open(VG_OUTPUT)))
    else:
        VG_STATUS = 0
    
    if SCRIPT_STATUS or VG_STATUS:
        print >>output, "******************** TEST '%s' FAILED! ********************"%(TESTNAME,)
        print >>output, "Command: "
        output.writelines(scriptLines)
        if not SCRIPT_STATUS:
            print >>output, "Output:"
        else:
            print >>output, "Incorrect Output:"
        cat(OUTPUT, output)
        if VG_STATUS:
            print >>output, "Valgrind Output:"
            cat(VG_OUTPUT, output)
        print >>output, "******************** TEST '%s' FAILED! ********************"%(TESTNAME,)
        output.flush()
        if xfailLines:
            return TestStatus.XPass
        else:
            return TestStatus.Fail

    if xfailLines:
        return TestStatus.XFail
    else:
        return TestStatus.Pass
    
def main():
    _,path = sys.argv
    command = path
    # Use hand concatentation here because we want to override
    # absolute paths.
    output = 'Output/' + path + '.out'
    testname = path

    # Determine which clang to use.
    CLANG = os.getenv('CLANG')
    if not CLANG:
        CLANG = 'clang'

    res = runOneTest(path, command, output, testname, CLANG,
                     useValgrind=bool(os.getenv('VG')), 
                     useDGCompat=bool(os.getenv('DG_COMPAT')), 
                     useScript=os.getenv("TEST_SCRIPT"))

    sys.exit(res == TestStatus.Fail or res == TestStatus.XPass)

if __name__=='__main__':
    main()
