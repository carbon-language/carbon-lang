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

import errno
import os
import re
import signal
import subprocess
import sys

# Increase determinism for things that use the terminal width.
#
# FIXME: Find a better place for this hack.
os.environ['COLUMNS'] = '0'

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

def runOneTest(FILENAME, SUBST, OUTPUT, TESTNAME, CLANG, CLANGCC,
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
                     (' clang ', ' ' + CLANG + ' '),
                     (' clang-cc ', ' ' + CLANGCC + ' ')]
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

        # Detect Ctrl-C in subprocess.
        if SCRIPT_STATUS == -signal.SIGINT:
            raise KeyboardInterrupt
    except KeyboardInterrupt:
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

def capture(args):
    p = subprocess.Popen(args, stdout=subprocess.PIPE)
    out,_ = p.communicate()
    return out

def which(command):
    # Would be nice if Python had a lib function for this.
    res = capture(['which',command])
    res = res.strip()
    if res and os.path.exists(res):
        return res
    return None

def inferClang():
    # Determine which clang to use.
    clang = os.getenv('CLANG')
    
    # If the user set clang in the environment, definitely use that and don't
    # try to validate.
    if clang:
        return clang

    # Otherwise look in the path.
    clang = which('clang')

    if not clang:
        print >>sys.stderr, "error: couldn't find 'clang' program, try setting CLANG in your environment"
        sys.exit(1)
        
    return clang

def inferClangCC(clang):
    clangcc = os.getenv('CLANGCC')

    # If the user set clang in the environment, definitely use that and don't
    # try to validate.
    if clangcc:
        return clangcc

    # Otherwise try adding -cc since we expect to be looking in a build
    # directory.
    clangcc = which(clang + '-cc')
    if not clangcc:
        # Otherwise ask clang.
        res = capture([clang, '-print-prog-name=clang-cc'])
        res = res.strip()
        if res and os.path.exists(res):
            clangcc = res
    
    if not clangcc:
        print >>sys.stderr, "error: couldn't find 'clang-cc' program, try setting CLANGCC in your environment"
        sys.exit(1)
        
    return clangcc
    
def main():
    global options
    from optparse import OptionParser
    parser = OptionParser("usage: %prog [options] {tests}")
    parser.add_option("", "--clang", dest="clang",
                      help="Program to use as \"clang\"",
                      action="store", default=None)
    parser.add_option("", "--clang-cc", dest="clangcc",
                      help="Program to use as \"clang-cc\"",
                      action="store", default=None)
    parser.add_option("", "--vg", dest="useValgrind",
                      help="Run tests under valgrind",
                      action="store_true", default=False)
    parser.add_option("", "--dg", dest="useDGCompat",
                      help="Use llvm dejagnu compatibility mode",
                      action="store_true", default=False)
    (opts, args) = parser.parse_args()

    if not args:
        parser.error('No tests specified')

    if opts.clang is None:
        opts.clang = inferClang()
    if opts.clangcc is None:
        opts.clangcc = inferClangCC(opts.clang)

    for path in args:
        command = path
        # Use hand concatentation here because we want to override
        # absolute paths.
        output = 'Output/' + path + '.out'
        testname = path
        
        res = runOneTest(path, command, output, testname, 
                         opts.clang, opts.clangcc,
                         useValgrind=opts.useValgrind,
                         useDGCompat=opts.useDGCompat,
                         useScript=os.getenv("TEST_SCRIPT"))

    sys.exit(res == TestStatus.Fail or res == TestStatus.XPass)

if __name__=='__main__':
    main()
