#!/usr/bin/env python
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
import platform
import re
import signal
import subprocess
import sys

import Util

kSystemName = platform.system()

class TestStatus:
    Pass = 0 
    XFail = 1
    Fail = 2
    XPass = 3
    Invalid = 4

    kNames = ['Pass','XFail','Fail','XPass','Invalid']
    @staticmethod
    def getName(code): 
        return TestStatus.kNames[code]

def executeScript(cfg, script, commands, cwd, useValgrind):
    # Write script file
    f = open(script,'w')
    if kSystemName == 'Windows':
        f.write('\nif %ERRORLEVEL% NEQ 0 EXIT\n'.join(commands))
    else:
        f.write(' &&\n'.join(commands))
    f.write('\n')
    f.close()

    if kSystemName == 'Windows':
        command = ['cmd','/c', script]
    else:
        command = ['/bin/sh', script]
        if useValgrind:
            # FIXME: Running valgrind on sh is overkill. We probably could just
            # ron on clang with no real loss.
            command = ['valgrind', '-q',
                       '--tool=memcheck', '--leak-check=no', '--trace-children=yes',
                       '--error-exitcode=123'] + command

    p = subprocess.Popen(command, cwd=cwd,
                         stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE,
                         env=cfg.environment)
    out,err = p.communicate()
    exitCode = p.wait()

    # Detect Ctrl-C in subprocess.
    if exitCode == -signal.SIGINT:
        raise KeyboardInterrupt

    return out, err, exitCode

import StringIO
def runOneTest(cfg, testPath, tmpBase, clang, clangcc, useValgrind):
    # Make paths absolute.
    tmpBase = os.path.abspath(tmpBase)
    testPath = os.path.abspath(testPath)

    # Create the output directory if it does not already exist.

    Util.mkdir_p(os.path.dirname(tmpBase))
    script = tmpBase + '.script'
    if kSystemName == 'Windows':
        script += '.bat'

    substitutions = [('%s', testPath),
                     ('%S', os.path.dirname(testPath)),
                     ('%t', tmpBase + '.tmp'),
                     (' clang ', ' ' + clang + ' '),
                     (' clang-cc ', ' ' + clangcc + ' ')]

    # Collect the test lines from the script.
    scriptLines = []
    xfailLines = []
    for ln in open(testPath):
        if 'RUN:' in ln:
            # Isolate the command to run.
            index = ln.index('RUN:')
            ln = ln[index+4:]
            
            # Strip trailing newline.
            scriptLines.append(ln)
        elif 'XFAIL' in ln:
            xfailLines.append(ln)
        
        # FIXME: Support something like END, in case we need to process large
        # files.

    # Verify the script contains a run line.
    if not scriptLines:
        return (TestStatus.Fail, "Test has no run line!")
    
    # Apply substitutions to the script.
    def processLine(ln):
        # Apply substitutions
        for a,b in substitutions:
            ln = ln.replace(a,b)

        # Strip the trailing newline and any extra whitespace.
        return ln.strip()
    scriptLines = map(processLine, scriptLines)    

    # Validate interior lines for '&&', a lovely historical artifact.
    for i in range(len(scriptLines) - 1):
        ln = scriptLines[i]

        if not ln.endswith('&&'):
            return (TestStatus.Fail, 
                    ("MISSING \'&&\': %s\n"  +
                     "FOLLOWED BY   : %s\n") % (ln, scriptLines[i + 1]))
    
        # Strip off '&&'
        scriptLines[i] = ln[:-2]

    out, err, exitCode = executeScript(cfg, script, scriptLines, 
                                       os.path.dirname(testPath),
                                       useValgrind)
    if xfailLines:
        ok = exitCode != 0
        status = (TestStatus.XPass, TestStatus.XFail)[ok]
    else:
        ok = exitCode == 0
        status = (TestStatus.Fail, TestStatus.Pass)[ok]

    if ok:
        return (status,'')

    output = StringIO.StringIO()
    print >>output, "Script:"
    print >>output, "--"
    print >>output, '\n'.join(scriptLines)
    print >>output, "--"
    print >>output, "Exit Code: %r" % exitCode
    print >>output, "Command Output (stdout):"
    print >>output, "--"
    output.write(out)
    print >>output, "--"
    print >>output, "Command Output (stderr):"
    print >>output, "--"
    output.write(err)
    print >>output, "--"
    return (status, output.getvalue())

def capture(args):
    p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out,_ = p.communicate()
    return out

def inferClang(cfg):
    # Determine which clang to use.
    clang = os.getenv('CLANG')
    
    # If the user set clang in the environment, definitely use that and don't
    # try to validate.
    if clang:
        return clang

    # Otherwise look in the path.
    clang = Util.which('clang', cfg.environment['PATH'])

    if not clang:
        print >>sys.stderr, "error: couldn't find 'clang' program, try setting CLANG in your environment"
        sys.exit(1)
        
    return clang

def inferClangCC(cfg, clang):
    clangcc = os.getenv('CLANGCC')

    # If the user set clang in the environment, definitely use that and don't
    # try to validate.
    if clangcc:
        return clangcc

    # Otherwise try adding -cc since we expect to be looking in a build
    # directory.
    if clang.endswith('.exe'):
        clangccName = clang[:-4] + '-cc.exe'
    else:
        clangccName = clang + '-cc'
    clangcc = Util.which(clangccName, cfg.environment['PATH'])
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
    
def getTestOutputBase(dir, testpath):
    """getTestOutputBase(dir, testpath) - Get the full path for temporary files
    corresponding to the given test path."""

    # Form the output base out of the test parent directory name and the test
    # name. FIXME: Find a better way to organize test results.
    return os.path.join(dir, 
                        os.path.basename(os.path.dirname(testpath)),
                        os.path.basename(testpath))
