import os
import platform
import re
import signal
import subprocess
import sys

import ShUtil
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

def executeShCmd(cmd, cfg, cwd, results):
    if isinstance(cmd, ShUtil.Seq):
        if cmd.op == ';':
            res = executeShCmd(cmd.lhs, cfg, cwd, results)
            return executeShCmd(cmd.rhs, cfg, cwd, results)

        if cmd.op == '&':
            raise NotImplementedError,"unsupported test command: '&'"

        if cmd.op == '||':
            res = executeShCmd(cmd.lhs, cfg, cwd, results)
            if res != 0:
                res = executeShCmd(cmd.rhs, cfg, cwd, results)
            return res
        if cmd.op == '&&':
            res = executeShCmd(cmd.lhs, cfg, cwd, results)
            if res is None:
                return res

            if res == 0:
                res = executeShCmd(cmd.rhs, cfg, cwd, results)
            return res
            
        raise ValueError,'Unknown shell command: %r' % cmd.op

    assert isinstance(cmd, ShUtil.Pipeline)
    procs = []
    input = subprocess.PIPE
    for j in cmd.commands:
        # FIXME: This is broken, it doesn't account for the accumulative nature
        # of redirects.
        stdin = input
        stdout = stderr = subprocess.PIPE
        for r in j.redirects:
            if r[0] == ('>',2):
                stderr = open(r[1], 'w')
            elif r[0] == ('>&',2) and r[1] == '1':
                stderr = subprocess.STDOUT
            elif r[0] == ('>',):
                stdout = open(r[1], 'w')
            elif r[0] == ('<',):
                stdin = open(r[1], 'r')
            else:
                raise NotImplementedError,"Unsupported redirect: %r" % r

        procs.append(subprocess.Popen(j.args, cwd=cwd,
                                      stdin = stdin,
                                      stdout = stdout,
                                      stderr = stderr,
                                      env = cfg.environment))

        # Immediately close stdin for any process taking stdin from us.
        if stdin == subprocess.PIPE:
            procs[-1].stdin.close()
            procs[-1].stdin = None

        if stdout == subprocess.PIPE:
            input = procs[-1].stdout
        else:
            input = subprocess.PIPE
    
    # FIXME: There is a potential for deadlock here, when we have a pipe and
    # some process other than the last one ends up blocked on stderr.
    procData = [None] * len(procs)
    procData[-1] = procs[-1].communicate()
    for i in range(len(procs) - 1):
        if procs[i].stdout is not None:
            out = procs[i].stdout.read()
        else:
            out = ''
        if procs[i].stderr is not None:
            err = procs[i].stderr.read()
        else:
            err = ''
        procData[i] = (out,err)

    # FIXME: Fix tests to work with pipefail, and make exitCode max across
    # procs.
    for i,(out,err) in enumerate(procData):
        exitCode = res = procs[i].wait()
        results.append((cmd.commands[i], out, err, res))

    if cmd.negate:
        exitCode = not exitCode

    return exitCode
        
def executeScriptInternal(cfg, commands, cwd):
    cmd = ShUtil.ShParser(' &&\n'.join(commands), 
                          kSystemName == 'Windows').parse()

    results = []
    try:
        exitCode = executeShCmd(cmd, cfg, cwd, results)
    except:
        import traceback

        out = ''
        err = 'Exception during script execution:\n%s\n' % traceback.format_exc()
        return out, err, 127

    out = err = ''
    for i,(cmd, cmd_out,cmd_err,res) in enumerate(results):
        out += 'Command %d: %s\n' % (i, ' '.join('"%s"' % s for s in cmd.args))
        out += 'Command %d Result: %r\n' % (i, res)
        out += 'Command %d Output:\n%s\n\n' % (i, cmd_out)
        out += 'Command %d Stderr:\n%s\n\n' % (i, cmd_err)

    return out, err, exitCode

def executeScript(cfg, script, commands, cwd):
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
        if cfg.useValgrind:
            # FIXME: Running valgrind on sh is overkill. We probably could just
            # run on clang with no real loss.
            valgrindArgs = ['valgrind', '-q',
                            '--tool=memcheck', '--trace-children=yes',
                            '--error-exitcode=123'] + cfg.valgrindArgs
            command = valgrindArgs + command

    p = subprocess.Popen(command, cwd=cwd,
                         stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE,
                         env=cfg.environment)
    out,err = p.communicate()
    exitCode = p.wait()

    return out, err, exitCode

import StringIO
def runOneTest(cfg, testPath, tmpBase):
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
                     (' clang ', ' ' + cfg.clang + ' '),
                     (' clang-cc ', ' ' + cfg.clangcc + ' ')]

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

    if not cfg.useExternalShell:
        res = executeScriptInternal(cfg, scriptLines, os.path.dirname(testPath))

        if res is not None:
            out, err, exitCode = res
        elif True:
            return (TestStatus.Fail, 
                    "Unable to execute internally:\n%s\n" 
                    % '\n'.join(scriptLines))
        else:
            out, err, exitCode = executeScript(cfg, script, scriptLines, 
                                               os.path.dirname(testPath))
    else:
        out, err, exitCode = executeScript(cfg, script, scriptLines, 
                                           os.path.dirname(testPath))

    # Detect Ctrl-C in subprocess.
    if exitCode == -signal.SIGINT:
        raise KeyboardInterrupt

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
