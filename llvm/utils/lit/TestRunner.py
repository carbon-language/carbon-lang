import os, signal, subprocess, sys
import StringIO

import ShUtil
import Test
import Util

class InternalShellError(Exception):
    def __init__(self, command, message):
        self.command = command
        self.message = message

def executeCommand(command, cwd=None, env=None):
    p = subprocess.Popen(command, cwd=cwd,
                         stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE,
                         env=env)
    out,err = p.communicate()
    exitCode = p.wait()

    # Detect Ctrl-C in subprocess.
    if exitCode == -signal.SIGINT:
        raise KeyboardInterrupt

    return out, err, exitCode

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
        redirects = [(0,), (1,), (2,)]
        for r in j.redirects:
            if r[0] == ('>',2):
                redirects[2] = [r[1], 'w', None]
            elif r[0] == ('>&',2) and r[1] in '012':
                redirects[2] = redirects[int(r[1])]
            elif r[0] == ('>&',) or r[0] == ('&>',):
                redirects[1] = redirects[2] = [r[1], 'w', None]
            elif r[0] == ('>',):
                redirects[1] = [r[1], 'w', None]
            elif r[0] == ('<',):
                redirects[0] = [r[1], 'r', None]
            else:
                raise NotImplementedError,"Unsupported redirect: %r" % (r,)

        final_redirects = []
        for index,r in enumerate(redirects):
            if r == (0,):
                result = input
            elif r == (1,):
                if index == 0:
                    raise NotImplementedError,"Unsupported redirect for stdin"
                elif index == 1:
                    result = subprocess.PIPE
                else:
                    result = subprocess.STDOUT
            elif r == (2,):
                if index != 2:
                    raise NotImplementedError,"Unsupported redirect on stdout"
                result = subprocess.PIPE
            else:
                if r[2] is None:
                    r[2] = open(r[0], r[1])
                result = r[2]
            final_redirects.append(result)

        stdin, stdout, stderr = final_redirects

        # If stderr wants to come from stdout, but stdout isn't a pipe, then put
        # stderr on a pipe and treat it as stdout.
        if (stderr == subprocess.STDOUT and stdout != subprocess.PIPE):
            stderr = subprocess.PIPE
            stderrIsStdout = True
        else:
            stderrIsStdout = False

        # Resolve the executable path ourselves.
        args = list(j.args)
        args[0] = Util.which(args[0], cfg.environment['PATH'])
        if not args[0]:
            raise InternalShellError(j, '%r: command not found' % j.args[0])

        procs.append(subprocess.Popen(j.args, cwd=cwd,
                                      stdin = stdin,
                                      stdout = stdout,
                                      stderr = stderr,
                                      env = cfg.environment,
                                      close_fds = True))

        # Immediately close stdin for any process taking stdin from us.
        if stdin == subprocess.PIPE:
            procs[-1].stdin.close()
            procs[-1].stdin = None

        # Update the current stdin source.
        if stdout == subprocess.PIPE:
            input = procs[-1].stdout
        elif stderrIsStdout:
            input = procs[-1].stderr
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

    exitCode = None
    for i,(out,err) in enumerate(procData):
        res = procs[i].wait()
        # Detect Ctrl-C in subprocess.
        if res == -signal.SIGINT:
            raise KeyboardInterrupt

        results.append((cmd.commands[i], out, err, res))
        if cmd.pipe_err:
            # Python treats the exit code as a signed char.
            if res < 0:
                exitCode = min(exitCode, res)
            else:
                exitCode = max(exitCode, res)
        else:
            exitCode = res

    if cmd.negate:
        exitCode = not exitCode

    return exitCode

def executeScriptInternal(test, litConfig, tmpBase, commands, cwd):
    ln = ' &&\n'.join(commands)
    try:
        cmd = ShUtil.ShParser(ln, litConfig.isWindows).parse()
    except:
        return (Test.FAIL, "shell parser error on: %r" % ln)

    results = []
    try:
        exitCode = executeShCmd(cmd, test.config, cwd, results)
    except InternalShellError,e:
        out = ''
        err = e.message
        exitCode = 255

    out = err = ''
    for i,(cmd, cmd_out,cmd_err,res) in enumerate(results):
        out += 'Command %d: %s\n' % (i, ' '.join('"%s"' % s for s in cmd.args))
        out += 'Command %d Result: %r\n' % (i, res)
        out += 'Command %d Output:\n%s\n\n' % (i, cmd_out)
        out += 'Command %d Stderr:\n%s\n\n' % (i, cmd_err)

    return out, err, exitCode

def executeTclScriptInternal(test, litConfig, tmpBase, commands, cwd):
    import TclUtil
    cmds = []
    for ln in commands:
        # Given the unfortunate way LLVM's test are written, the line gets
        # backslash substitution done twice.
        ln = TclUtil.TclLexer(ln).lex_unquoted(process_all = True)

        try:
            tokens = list(TclUtil.TclLexer(ln).lex())
        except:
            return (Test.FAIL, "Tcl lexer error on: %r" % ln)

        # Validate there are no control tokens.
        for t in tokens:
            if not isinstance(t, str):
                return (Test.FAIL,
                        "Invalid test line: %r containing %r" % (ln, t))

        try:
            cmds.append(TclUtil.TclExecCommand(tokens).parse_pipeline())
        except:
            return (Test.FAIL, "Tcl 'exec' parse error on: %r" % ln)

    cmd = cmds[0]
    for c in cmds[1:]:
        cmd = ShUtil.Seq(cmd, '&&', c)

    if litConfig.useTclAsSh:
        script = tmpBase + '.script'

        # Write script file
        f = open(script,'w')
        print >>f, 'set -o pipefail'
        cmd.toShell(f, pipefail = True)
        f.close()

        if 0:
            print >>sys.stdout, cmd
            print >>sys.stdout, open(script).read()
            print >>sys.stdout
            return '', '', 0

        command = ['/bin/bash', script]
        out,err,exitCode = executeCommand(command, cwd=cwd,
                                          env=test.config.environment)

        # Tcl commands fail on standard error output.
        if err:
            exitCode = 1
            out = 'Command has output on stderr!\n\n' + out

        return out,err,exitCode
    else:
        results = []
        try:
            exitCode = executeShCmd(cmd, test.config, cwd, results)
        except InternalShellError,e:
            results.append((e.command, '', e.message + '\n', 255))
            exitCode = 255

    out = err = ''

    # Tcl commands fail on standard error output.
    if [True for _,_,err,res in results if err]:
        exitCode = 1
        out += 'Command has output on stderr!\n\n'

    for i,(cmd, cmd_out, cmd_err, res) in enumerate(results):
        out += 'Command %d: %s\n' % (i, ' '.join('"%s"' % s for s in cmd.args))
        out += 'Command %d Result: %r\n' % (i, res)
        out += 'Command %d Output:\n%s\n\n' % (i, cmd_out)
        out += 'Command %d Stderr:\n%s\n\n' % (i, cmd_err)

    return out, err, exitCode

def executeScript(test, litConfig, tmpBase, commands, cwd):
    script = tmpBase + '.script'
    if litConfig.isWindows:
        script += '.bat'

    # Write script file
    f = open(script,'w')
    if litConfig.isWindows:
        f.write('\nif %ERRORLEVEL% NEQ 0 EXIT\n'.join(commands))
    else:
        f.write(' &&\n'.join(commands))
    f.write('\n')
    f.close()

    if litConfig.isWindows:
        command = ['cmd','/c', script]
    else:
        command = ['/bin/sh', script]
        if litConfig.useValgrind:
            # FIXME: Running valgrind on sh is overkill. We probably could just
            # run on clang with no real loss.
            valgrindArgs = ['valgrind', '-q',
                            '--tool=memcheck', '--trace-children=yes',
                            '--error-exitcode=123']
            valgrindArgs.extend(litConfig.valgrindArgs)

            command = valgrindArgs + command

    return executeCommand(command, cwd=cwd, env=test.config.environment)

def parseIntegratedTestScript(test, xfailHasColon, requireAndAnd):
    """parseIntegratedTestScript - Scan an LLVM/Clang style integrated test
    script and extract the lines to 'RUN' as well as 'XFAIL' and 'XTARGET'
    information. The RUN lines also will have variable substitution performed.
    """

    # Get the temporary location, this is always relative to the test suite
    # root, not test source root.
    #
    # FIXME: This should not be here?
    sourcepath = test.getSourcePath()
    execpath = test.getExecPath()
    execdir,execbase = os.path.split(execpath)
    tmpBase = os.path.join(execdir, 'Output', execbase)

    # We use #_MARKER_# to hide %% while we do the other substitutions.
    substitutions = [('%%', '#_MARKER_#')]
    substitutions.extend(test.config.substitutions)
    substitutions.extend([('%s', sourcepath),
                          ('%S', os.path.dirname(sourcepath)),
                          ('%p', os.path.dirname(sourcepath)),
                          ('%t', tmpBase + '.tmp'),
                          # FIXME: Remove this once we kill DejaGNU.
                          ('%abs_tmp', tmpBase + '.tmp'),
                          ('#_MARKER_#', '%')])

    # Collect the test lines from the script.
    script = []
    xfails = []
    xtargets = []
    for ln in open(sourcepath):
        if 'RUN:' in ln:
            # Isolate the command to run.
            index = ln.index('RUN:')
            ln = ln[index+4:]

            # Trim trailing whitespace.
            ln = ln.rstrip()

            # Collapse lines with trailing '\\'.
            if script and script[-1][-1] == '\\':
                script[-1] = script[-1][:-1] + ln
            else:
                script.append(ln)
        elif xfailHasColon and 'XFAIL:' in ln:
            items = ln[ln.index('XFAIL:') + 6:].split(',')
            xfails.extend([s.strip() for s in items])
        elif not xfailHasColon and 'XFAIL' in ln:
            items = ln[ln.index('XFAIL') + 5:].split(',')
            xfails.extend([s.strip() for s in items])
        elif 'XTARGET:' in ln:
            items = ln[ln.index('XTARGET:') + 8:].split(',')
            xtargets.extend([s.strip() for s in items])
        elif 'END.' in ln:
            # Check for END. lines.
            if ln[ln.index('END.'):].strip() == 'END.':
                break

    # Apply substitutions to the script.
    def processLine(ln):
        # Apply substitutions
        for a,b in substitutions:
            ln = ln.replace(a,b)

        # Strip the trailing newline and any extra whitespace.
        return ln.strip()
    script = map(processLine, script)

    # Verify the script contains a run line.
    if not script:
        return (Test.UNRESOLVED, "Test has no run line!")

    if script[-1][-1] == '\\':
        return (Test.UNRESOLVED, "Test has unterminated run lines (with '\\')")

    # Validate interior lines for '&&', a lovely historical artifact.
    if requireAndAnd:
        for i in range(len(script) - 1):
            ln = script[i]

            if not ln.endswith('&&'):
                return (Test.FAIL,
                        ("MISSING \'&&\': %s\n"  +
                         "FOLLOWED BY   : %s\n") % (ln, script[i + 1]))

            # Strip off '&&'
            script[i] = ln[:-2]

    return script,xfails,xtargets,tmpBase,execdir

def formatTestOutput(status, out, err, exitCode, script):
    output = StringIO.StringIO()
    print >>output, "Script:"
    print >>output, "--"
    print >>output, '\n'.join(script)
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

def executeTclTest(test, litConfig):
    if test.config.unsupported:
        return (Test.UNSUPPORTED, 'Test is unsupported')

    res = parseIntegratedTestScript(test, True, False)
    if len(res) == 2:
        return res

    script, xfails, xtargets, tmpBase, execdir = res

    if litConfig.noExecute:
        return (Test.PASS, '')

    # Create the output directory if it does not already exist.
    Util.mkdir_p(os.path.dirname(tmpBase))

    res = executeTclScriptInternal(test, litConfig, tmpBase, script, execdir)
    if len(res) == 2:
        return res

    isXFail = False
    for item in xfails:
        if item == '*' or item in test.suite.config.target_triple:
            isXFail = True
            break

    # If this is XFAIL, see if it is expected to pass on this target.
    if isXFail:
        for item in xtargets:
            if item == '*' or item in test.suite.config.target_triple:
                isXFail = False
                break

    out,err,exitCode = res
    if isXFail:
        ok = exitCode != 0
        status = (Test.XPASS, Test.XFAIL)[ok]
    else:
        ok = exitCode == 0
        status = (Test.FAIL, Test.PASS)[ok]

    if ok:
        return (status,'')

    return formatTestOutput(status, out, err, exitCode, script)

def executeShTest(test, litConfig, useExternalSh, requireAndAnd):
    if test.config.unsupported:
        return (Test.UNSUPPORTED, 'Test is unsupported')

    res = parseIntegratedTestScript(test, False, requireAndAnd)
    if len(res) == 2:
        return res

    script, xfails, xtargets, tmpBase, execdir = res

    if litConfig.noExecute:
        return (Test.PASS, '')

    # Create the output directory if it does not already exist.
    Util.mkdir_p(os.path.dirname(tmpBase))

    if useExternalSh:
        res = executeScript(test, litConfig, tmpBase, script, execdir)
    else:
        res = executeScriptInternal(test, litConfig, tmpBase, script, execdir)
    if len(res) == 2:
        return res

    out,err,exitCode = res
    if xfails:
        ok = exitCode != 0
        status = (Test.XPASS, Test.XFAIL)[ok]
    else:
        ok = exitCode == 0
        status = (Test.FAIL, Test.PASS)[ok]

    if ok:
        return (status,'')

    return formatTestOutput(status, out, err, exitCode, script)
