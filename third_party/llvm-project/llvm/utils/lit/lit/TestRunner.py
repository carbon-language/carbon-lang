from __future__ import absolute_import
import errno
import io
import itertools
import getopt
import os, signal, subprocess, sys
import re
import stat
import platform
import shutil
import tempfile
import threading

import io
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO

from lit.ShCommands import GlobItem, Command
import lit.ShUtil as ShUtil
import lit.Test as Test
import lit.util
from lit.util import to_bytes, to_string, to_unicode
from lit.BooleanExpression import BooleanExpression

class InternalShellError(Exception):
    def __init__(self, command, message):
        self.command = command
        self.message = message

kIsWindows = platform.system() == 'Windows'

# Don't use close_fds on Windows.
kUseCloseFDs = not kIsWindows

# Use temporary files to replace /dev/null on Windows.
kAvoidDevNull = kIsWindows
kDevNull = "/dev/null"

# A regex that matches %dbg(ARG), which lit inserts at the beginning of each
# run command pipeline such that ARG specifies the pipeline's source line
# number.  lit later expands each %dbg(ARG) to a command that behaves as a null
# command in the target shell so that the line number is seen in lit's verbose
# mode.
#
# This regex captures ARG.  ARG must not contain a right parenthesis, which
# terminates %dbg.  ARG must not contain quotes, in which ARG might be enclosed
# during expansion.
kPdbgRegex = '%dbg\\(([^)\'"]*)\\)'

class ShellEnvironment(object):

    """Mutable shell environment containing things like CWD and env vars.

    Environment variables are not implemented, but cwd tracking is.
    """

    def __init__(self, cwd, env):
        self.cwd = cwd
        self.env = dict(env)

class TimeoutHelper(object):
    """
        Object used to helper manage enforcing a timeout in
        _executeShCmd(). It is passed through recursive calls
        to collect processes that have been executed so that when
        the timeout happens they can be killed.
    """
    def __init__(self, timeout):
        self.timeout = timeout
        self._procs = []
        self._timeoutReached = False
        self._doneKillPass = False
        # This lock will be used to protect concurrent access
        # to _procs and _doneKillPass
        self._lock = None
        self._timer = None

    def cancel(self):
        if not self.active():
            return
        self._timer.cancel()

    def active(self):
        return self.timeout > 0

    def addProcess(self, proc):
        if not self.active():
            return
        needToRunKill = False
        with self._lock:
            self._procs.append(proc)
            # Avoid re-entering the lock by finding out if kill needs to be run
            # again here but call it if necessary once we have left the lock.
            # We could use a reentrant lock here instead but this code seems
            # clearer to me.
            needToRunKill = self._doneKillPass

        # The initial call to _kill() from the timer thread already happened so
        # we need to call it again from this thread, otherwise this process
        # will be left to run even though the timeout was already hit
        if needToRunKill:
            assert self.timeoutReached()
            self._kill()

    def startTimer(self):
        if not self.active():
            return

        # Do some late initialisation that's only needed
        # if there is a timeout set
        self._lock = threading.Lock()
        self._timer = threading.Timer(self.timeout, self._handleTimeoutReached)
        self._timer.start()

    def _handleTimeoutReached(self):
        self._timeoutReached = True
        self._kill()

    def timeoutReached(self):
        return self._timeoutReached

    def _kill(self):
        """
            This method may be called multiple times as we might get unlucky
            and be in the middle of creating a new process in _executeShCmd()
            which won't yet be in ``self._procs``. By locking here and in
            addProcess() we should be able to kill processes launched after
            the initial call to _kill()
        """
        with self._lock:
            for p in self._procs:
                lit.util.killProcessAndChildren(p.pid)
            # Empty the list and note that we've done a pass over the list
            self._procs = [] # Python2 doesn't have list.clear()
            self._doneKillPass = True

class ShellCommandResult(object):
    """Captures the result of an individual command."""

    def __init__(self, command, stdout, stderr, exitCode, timeoutReached,
                 outputFiles = []):
        self.command = command
        self.stdout = stdout
        self.stderr = stderr
        self.exitCode = exitCode
        self.timeoutReached = timeoutReached
        self.outputFiles = list(outputFiles)

def executeShCmd(cmd, shenv, results, timeout=0):
    """
        Wrapper around _executeShCmd that handles
        timeout
    """
    # Use the helper even when no timeout is required to make
    # other code simpler (i.e. avoid bunch of ``!= None`` checks)
    timeoutHelper = TimeoutHelper(timeout)
    if timeout > 0:
        timeoutHelper.startTimer()
    finalExitCode = _executeShCmd(cmd, shenv, results, timeoutHelper)
    timeoutHelper.cancel()
    timeoutInfo = None
    if timeoutHelper.timeoutReached():
        timeoutInfo = 'Reached timeout of {} seconds'.format(timeout)

    return (finalExitCode, timeoutInfo)

def expand_glob(arg, cwd):
    if isinstance(arg, GlobItem):
        return sorted(arg.resolve(cwd))
    return [arg]

def expand_glob_expressions(args, cwd):
    result = [args[0]]
    for arg in args[1:]:
        result.extend(expand_glob(arg, cwd))
    return result

def quote_windows_command(seq):
    """
    Reimplement Python's private subprocess.list2cmdline for MSys compatibility

    Based on CPython implementation here:
      https://hg.python.org/cpython/file/849826a900d2/Lib/subprocess.py#l422

    Some core util distributions (MSys) don't tokenize command line arguments
    the same way that MSVC CRT does. Lit rolls its own quoting logic similar to
    the stock CPython logic to paper over these quoting and tokenization rule
    differences.

    We use the same algorithm from MSDN as CPython
    (http://msdn.microsoft.com/en-us/library/17w5ykft.aspx), but we treat more
    characters as needing quoting, such as double quotes themselves, and square
    brackets.

    For MSys based tools, this is very brittle though, because quoting an
    argument makes the MSys based tool unescape backslashes where it shouldn't
    (e.g. "a\b\\c\\\\d" becomes "a\b\c\\d" where it should stay as it was,
    according to regular win32 command line parsing rules).
    """
    result = []
    needquote = False
    for arg in seq:
        bs_buf = []

        # Add a space to separate this argument from the others
        if result:
            result.append(' ')

        # This logic differs from upstream list2cmdline.
        needquote = (" " in arg) or ("\t" in arg) or ("\"" in arg) or ("[" in arg) or (";" in arg) or not arg
        if needquote:
            result.append('"')

        for c in arg:
            if c == '\\':
                # Don't know if we need to double yet.
                bs_buf.append(c)
            elif c == '"':
                # Double backslashes.
                result.append('\\' * len(bs_buf)*2)
                bs_buf = []
                result.append('\\"')
            else:
                # Normal char
                if bs_buf:
                    result.extend(bs_buf)
                    bs_buf = []
                result.append(c)

        # Add remaining backslashes, if any.
        if bs_buf:
            result.extend(bs_buf)

        if needquote:
            result.extend(bs_buf)
            result.append('"')

    return ''.join(result)

# args are from 'export' or 'env' command.
# Skips the command, and parses its arguments.
# Modifies env accordingly.
# Returns copy of args without the command or its arguments.
def updateEnv(env, args):
    arg_idx_next = len(args)
    unset_next_env_var = False
    for arg_idx, arg in enumerate(args[1:]):
        # Support for the -u flag (unsetting) for env command
        # e.g., env -u FOO -u BAR will remove both FOO and BAR
        # from the environment.
        if arg == '-u':
            unset_next_env_var = True
            continue
        if unset_next_env_var:
            unset_next_env_var = False
            if arg in env.env:
                del env.env[arg]
            continue

        # Partition the string into KEY=VALUE.
        key, eq, val = arg.partition('=')
        # Stop if there was no equals.
        if eq == '':
            arg_idx_next = arg_idx + 1
            break
        env.env[key] = val
    return args[arg_idx_next:]

def executeBuiltinCd(cmd, shenv):
    """executeBuiltinCd - Change the current directory."""
    if len(cmd.args) != 2:
        raise InternalShellError("'cd' supports only one argument")
    newdir = cmd.args[1]
    # Update the cwd in the parent environment.
    if os.path.isabs(newdir):
        shenv.cwd = newdir
    else:
        shenv.cwd = os.path.realpath(os.path.join(shenv.cwd, newdir))
    # The cd builtin always succeeds. If the directory does not exist, the
    # following Popen calls will fail instead.
    return ShellCommandResult(cmd, "", "", 0, False)

def executeBuiltinExport(cmd, shenv):
    """executeBuiltinExport - Set an environment variable."""
    if len(cmd.args) != 2:
        raise InternalShellError("'export' supports only one argument")
    updateEnv(shenv, cmd.args)
    return ShellCommandResult(cmd, "", "", 0, False)

def executeBuiltinEcho(cmd, shenv):
    """Interpret a redirected echo command"""
    opened_files = []
    stdin, stdout, stderr = processRedirects(cmd, subprocess.PIPE, shenv,
                                             opened_files)
    if stdin != subprocess.PIPE or stderr != subprocess.PIPE:
        raise InternalShellError(
                cmd, "stdin and stderr redirects not supported for echo")

    # Some tests have un-redirected echo commands to help debug test failures.
    # Buffer our output and return it to the caller.
    is_redirected = True
    encode = lambda x : x
    if stdout == subprocess.PIPE:
        is_redirected = False
        stdout = StringIO()
    elif kIsWindows:
        # Reopen stdout in binary mode to avoid CRLF translation. The versions
        # of echo we are replacing on Windows all emit plain LF, and the LLVM
        # tests now depend on this.
        # When we open as binary, however, this also means that we have to write
        # 'bytes' objects to stdout instead of 'str' objects.
        encode = lit.util.to_bytes
        stdout = open(stdout.name, stdout.mode + 'b')
        opened_files.append((None, None, stdout, None))

    # Implement echo flags. We only support -e and -n, and not yet in
    # combination. We have to ignore unknown flags, because `echo "-D FOO"`
    # prints the dash.
    args = cmd.args[1:]
    interpret_escapes = False
    write_newline = True
    while len(args) >= 1 and args[0] in ('-e', '-n'):
        flag = args[0]
        args = args[1:]
        if flag == '-e':
            interpret_escapes = True
        elif flag == '-n':
            write_newline = False

    def maybeUnescape(arg):
        if not interpret_escapes:
            return arg

        arg = lit.util.to_bytes(arg)
        codec = 'string_escape' if sys.version_info < (3,0) else 'unicode_escape'
        return arg.decode(codec)

    if args:
        for arg in args[:-1]:
            stdout.write(encode(maybeUnescape(arg)))
            stdout.write(encode(' '))
        stdout.write(encode(maybeUnescape(args[-1])))
    if write_newline:
        stdout.write(encode('\n'))

    for (name, mode, f, path) in opened_files:
        f.close()

    output = "" if is_redirected else stdout.getvalue()
    return ShellCommandResult(cmd, output, "", 0, False)

def executeBuiltinMkdir(cmd, cmd_shenv):
    """executeBuiltinMkdir - Create new directories."""
    args = expand_glob_expressions(cmd.args, cmd_shenv.cwd)[1:]
    try:
        opts, args = getopt.gnu_getopt(args, 'p')
    except getopt.GetoptError as err:
        raise InternalShellError(cmd, "Unsupported: 'mkdir':  %s" % str(err))

    parent = False
    for o, a in opts:
        if o == "-p":
            parent = True
        else:
            assert False, "unhandled option"

    if len(args) == 0:
        raise InternalShellError(cmd, "Error: 'mkdir' is missing an operand")

    stderr = StringIO()
    exitCode = 0
    for dir in args:
        cwd = cmd_shenv.cwd
        dir = to_unicode(dir) if kIsWindows else to_bytes(dir)
        cwd = to_unicode(cwd) if kIsWindows else to_bytes(cwd)
        if not os.path.isabs(dir):
            dir = os.path.realpath(os.path.join(cwd, dir))
        if parent:
            lit.util.mkdir_p(dir)
        else:
            try:
                lit.util.mkdir(dir)
            except OSError as err:
                stderr.write("Error: 'mkdir' command failed, %s\n" % str(err))
                exitCode = 1
    return ShellCommandResult(cmd, "", stderr.getvalue(), exitCode, False)

def executeBuiltinRm(cmd, cmd_shenv):
    """executeBuiltinRm - Removes (deletes) files or directories."""
    args = expand_glob_expressions(cmd.args, cmd_shenv.cwd)[1:]
    try:
        opts, args = getopt.gnu_getopt(args, "frR", ["--recursive"])
    except getopt.GetoptError as err:
        raise InternalShellError(cmd, "Unsupported: 'rm':  %s" % str(err))

    force = False
    recursive = False
    for o, a in opts:
        if o == "-f":
            force = True
        elif o in ("-r", "-R", "--recursive"):
            recursive = True
        else:
            assert False, "unhandled option"

    if len(args) == 0:
        raise InternalShellError(cmd, "Error: 'rm' is missing an operand")

    def on_rm_error(func, path, exc_info):
        # path contains the path of the file that couldn't be removed
        # let's just assume that it's read-only and remove it.
        os.chmod(path, stat.S_IMODE( os.stat(path).st_mode) | stat.S_IWRITE)
        os.remove(path)

    stderr = StringIO()
    exitCode = 0
    for path in args:
        cwd = cmd_shenv.cwd
        path = to_unicode(path) if kIsWindows else to_bytes(path)
        cwd = to_unicode(cwd) if kIsWindows else to_bytes(cwd)
        if not os.path.isabs(path):
            path = os.path.realpath(os.path.join(cwd, path))
        if force and not os.path.exists(path):
            continue
        try:
            if os.path.isdir(path):
                if not recursive:
                    stderr.write("Error: %s is a directory\n" % path)
                    exitCode = 1
                if platform.system() == 'Windows':
                    # NOTE: use ctypes to access `SHFileOperationsW` on Windows to
                    # use the NT style path to get access to long file paths which
                    # cannot be removed otherwise.
                    from ctypes.wintypes import BOOL, HWND, LPCWSTR, UINT, WORD
                    from ctypes import addressof, byref, c_void_p, create_unicode_buffer
                    from ctypes import Structure
                    from ctypes import windll, WinError, POINTER

                    class SHFILEOPSTRUCTW(Structure):
                        _fields_ = [
                                ('hWnd', HWND),
                                ('wFunc', UINT),
                                ('pFrom', LPCWSTR),
                                ('pTo', LPCWSTR),
                                ('fFlags', WORD),
                                ('fAnyOperationsAborted', BOOL),
                                ('hNameMappings', c_void_p),
                                ('lpszProgressTitle', LPCWSTR),
                        ]

                    FO_MOVE, FO_COPY, FO_DELETE, FO_RENAME = range(1, 5)

                    FOF_SILENT = 4
                    FOF_NOCONFIRMATION = 16
                    FOF_NOCONFIRMMKDIR = 512
                    FOF_NOERRORUI = 1024

                    FOF_NO_UI = FOF_SILENT | FOF_NOCONFIRMATION | FOF_NOERRORUI | FOF_NOCONFIRMMKDIR

                    SHFileOperationW = windll.shell32.SHFileOperationW
                    SHFileOperationW.argtypes = [POINTER(SHFILEOPSTRUCTW)]

                    path = os.path.abspath(path)

                    pFrom = create_unicode_buffer(path, len(path) + 2)
                    pFrom[len(path)] = pFrom[len(path) + 1] = '\0'
                    operation = SHFILEOPSTRUCTW(wFunc=UINT(FO_DELETE),
                                                pFrom=LPCWSTR(addressof(pFrom)),
                                                fFlags=FOF_NO_UI)
                    result = SHFileOperationW(byref(operation))
                    if result:
                        raise WinError(result)
                else:
                    shutil.rmtree(path, onerror = on_rm_error if force else None)
            else:
                if force and not os.access(path, os.W_OK):
                    os.chmod(path,
                             stat.S_IMODE(os.stat(path).st_mode) | stat.S_IWRITE)
                os.remove(path)
        except OSError as err:
            stderr.write("Error: 'rm' command failed, %s" % str(err))
            exitCode = 1
    return ShellCommandResult(cmd, "", stderr.getvalue(), exitCode, False)

def executeBuiltinColon(cmd, cmd_shenv):
    """executeBuiltinColon - Discard arguments and exit with status 0."""
    return ShellCommandResult(cmd, "", "", 0, False)

def processRedirects(cmd, stdin_source, cmd_shenv, opened_files):
    """Return the standard fds for cmd after applying redirects

    Returns the three standard file descriptors for the new child process.  Each
    fd may be an open, writable file object or a sentinel value from the
    subprocess module.
    """

    # Apply the redirections, we use (N,) as a sentinel to indicate stdin,
    # stdout, stderr for N equal to 0, 1, or 2 respectively. Redirects to or
    # from a file are represented with a list [file, mode, file-object]
    # where file-object is initially None.
    redirects = [(0,), (1,), (2,)]
    for (op, filename) in cmd.redirects:
        if op == ('>',2):
            redirects[2] = [filename, 'w', None]
        elif op == ('>>',2):
            redirects[2] = [filename, 'a', None]
        elif op == ('>&',2) and filename in '012':
            redirects[2] = redirects[int(filename)]
        elif op == ('>&',) or op == ('&>',):
            redirects[1] = redirects[2] = [filename, 'w', None]
        elif op == ('>',):
            redirects[1] = [filename, 'w', None]
        elif op == ('>>',):
            redirects[1] = [filename, 'a', None]
        elif op == ('<',):
            redirects[0] = [filename, 'r', None]
        else:
            raise InternalShellError(cmd, "Unsupported redirect: %r" % ((op, filename),))

    # Open file descriptors in a second pass.
    std_fds = [None, None, None]
    for (index, r) in enumerate(redirects):
        # Handle the sentinel values for defaults up front.
        if isinstance(r, tuple):
            if r == (0,):
                fd = stdin_source
            elif r == (1,):
                if index == 0:
                    raise InternalShellError(cmd, "Unsupported redirect for stdin")
                elif index == 1:
                    fd = subprocess.PIPE
                else:
                    fd = subprocess.STDOUT
            elif r == (2,):
                if index != 2:
                    raise InternalShellError(cmd, "Unsupported redirect on stdout")
                fd = subprocess.PIPE
            else:
                raise InternalShellError(cmd, "Bad redirect")
            std_fds[index] = fd
            continue

        (filename, mode, fd) = r

        # Check if we already have an open fd. This can happen if stdout and
        # stderr go to the same place.
        if fd is not None:
            std_fds[index] = fd
            continue

        redir_filename = None
        name = expand_glob(filename, cmd_shenv.cwd)
        if len(name) != 1:
           raise InternalShellError(cmd, "Unsupported: glob in "
                                    "redirect expanded to multiple files")
        name = name[0]
        if kAvoidDevNull and name == kDevNull:
            fd = tempfile.TemporaryFile(mode=mode)
        elif kIsWindows and name == '/dev/tty':
            # Simulate /dev/tty on Windows.
            # "CON" is a special filename for the console.
            fd = open("CON", mode)
        else:
            # Make sure relative paths are relative to the cwd.
            redir_filename = os.path.join(cmd_shenv.cwd, name)
            redir_filename = to_unicode(redir_filename) \
                    if kIsWindows else to_bytes(redir_filename)
            fd = open(redir_filename, mode)
        # Workaround a Win32 and/or subprocess bug when appending.
        #
        # FIXME: Actually, this is probably an instance of PR6753.
        if mode == 'a':
            fd.seek(0, 2)
        # Mutate the underlying redirect list so that we can redirect stdout
        # and stderr to the same place without opening the file twice.
        r[2] = fd
        opened_files.append((filename, mode, fd) + (redir_filename,))
        std_fds[index] = fd

    return std_fds

def _executeShCmd(cmd, shenv, results, timeoutHelper):
    if timeoutHelper.timeoutReached():
        # Prevent further recursion if the timeout has been hit
        # as we should try avoid launching more processes.
        return None

    if isinstance(cmd, ShUtil.Seq):
        if cmd.op == ';':
            res = _executeShCmd(cmd.lhs, shenv, results, timeoutHelper)
            return _executeShCmd(cmd.rhs, shenv, results, timeoutHelper)

        if cmd.op == '&':
            raise InternalShellError(cmd,"unsupported shell operator: '&'")

        if cmd.op == '||':
            res = _executeShCmd(cmd.lhs, shenv, results, timeoutHelper)
            if res != 0:
                res = _executeShCmd(cmd.rhs, shenv, results, timeoutHelper)
            return res

        if cmd.op == '&&':
            res = _executeShCmd(cmd.lhs, shenv, results, timeoutHelper)
            if res is None:
                return res

            if res == 0:
                res = _executeShCmd(cmd.rhs, shenv, results, timeoutHelper)
            return res

        raise ValueError('Unknown shell command: %r' % cmd.op)
    assert isinstance(cmd, ShUtil.Pipeline)

    procs = []
    proc_not_counts = []
    default_stdin = subprocess.PIPE
    stderrTempFiles = []
    opened_files = []
    named_temp_files = []
    builtin_commands = set(['cat', 'diff'])
    builtin_commands_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "builtin_commands")
    inproc_builtins = {'cd': executeBuiltinCd,
                       'export': executeBuiltinExport,
                       'echo': executeBuiltinEcho,
                       'mkdir': executeBuiltinMkdir,
                       'rm': executeBuiltinRm,
                       ':': executeBuiltinColon}
    # To avoid deadlock, we use a single stderr stream for piped
    # output. This is null until we have seen some output using
    # stderr.
    for i,j in enumerate(cmd.commands):
        # Reference the global environment by default.
        cmd_shenv = shenv
        args = list(j.args)
        not_args = []
        not_count = 0
        not_crash = False
        while True:
            if args[0] == 'env':
                # Create a copy of the global environment and modify it for
                # this one command. There might be multiple envs in a pipeline,
                # and there might be multiple envs in a command (usually when
                # one comes from a substitution):
                #   env FOO=1 llc < %s | env BAR=2 llvm-mc | FileCheck %s
                #   env FOO=1 %{another_env_plus_cmd} | FileCheck %s
                if cmd_shenv is shenv:
                    cmd_shenv = ShellEnvironment(shenv.cwd, shenv.env)
                args = updateEnv(cmd_shenv, args)
                if not args:
                    raise InternalShellError(j, "Error: 'env' requires a"
                                                " subcommand")
            elif args[0] == 'not':
                not_args.append(args.pop(0))
                not_count += 1
                if args and args[0] == '--crash':
                    not_args.append(args.pop(0))
                    not_crash = True
                if not args:
                    raise InternalShellError(j, "Error: 'not' requires a"
                                                " subcommand")
            elif args[0] == '!':
                not_args.append(args.pop(0))
                not_count += 1
                if not args:
                    raise InternalShellError(j, "Error: '!' requires a"
                                                " subcommand")
            else:
                break

        # Handle in-process builtins.
        #
        # Handle "echo" as a builtin if it is not part of a pipeline. This
        # greatly speeds up tests that construct input files by repeatedly
        # echo-appending to a file.
        # FIXME: Standardize on the builtin echo implementation. We can use a
        # temporary file to sidestep blocking pipe write issues.
        inproc_builtin = inproc_builtins.get(args[0], None)
        if inproc_builtin and (args[0] != 'echo' or len(cmd.commands) == 1):
            # env calling an in-process builtin is useless, so we take the safe
            # approach of complaining.
            if not cmd_shenv is shenv:
                raise InternalShellError(j, "Error: 'env' cannot call '{}'"
                                            .format(args[0]))
            if not_crash:
                raise InternalShellError(j, "Error: 'not --crash' cannot call"
                                            " '{}'".format(args[0]))
            if len(cmd.commands) != 1:
                raise InternalShellError(j, "Unsupported: '{}' cannot be part"
                                            " of a pipeline".format(args[0]))
            result = inproc_builtin(Command(args, j.redirects), cmd_shenv)
            if not_count % 2:
                result.exitCode = int(not result.exitCode)
            result.command.args = j.args;
            results.append(result)
            return result.exitCode

        # Resolve any out-of-process builtin command before adding back 'not'
        # commands.
        if args[0] in builtin_commands:
            args.insert(0, sys.executable)
            cmd_shenv.env['PYTHONPATH'] = \
                os.path.dirname(os.path.abspath(__file__))
            args[1] = os.path.join(builtin_commands_dir, args[1] + ".py")

        # We had to search through the 'not' commands to find all the 'env'
        # commands and any other in-process builtin command.  We don't want to
        # reimplement 'not' and its '--crash' here, so just push all 'not'
        # commands back to be called as external commands.  Because this
        # approach effectively moves all 'env' commands up front, it relies on
        # the assumptions that (1) environment variables are not intended to be
        # relevant to 'not' commands and (2) the 'env' command should always
        # blindly pass along the status it receives from any command it calls.

        # For plain negations, either 'not' without '--crash', or the shell
        # operator '!', leave them out from the command to execute and
        # invert the result code afterwards.
        if not_crash:
            args = not_args + args
            not_count = 0
        else:
            not_args = []

        stdin, stdout, stderr = processRedirects(j, default_stdin, cmd_shenv,
                                                 opened_files)

        # If stderr wants to come from stdout, but stdout isn't a pipe, then put
        # stderr on a pipe and treat it as stdout.
        if (stderr == subprocess.STDOUT and stdout != subprocess.PIPE):
            stderr = subprocess.PIPE
            stderrIsStdout = True
        else:
            stderrIsStdout = False

            # Don't allow stderr on a PIPE except for the last
            # process, this could deadlock.
            #
            # FIXME: This is slow, but so is deadlock.
            if stderr == subprocess.PIPE and j != cmd.commands[-1]:
                stderr = tempfile.TemporaryFile(mode='w+b')
                stderrTempFiles.append((i, stderr))

        # Resolve the executable path ourselves.
        executable = None
        # For paths relative to cwd, use the cwd of the shell environment.
        if args[0].startswith('.'):
            exe_in_cwd = os.path.join(cmd_shenv.cwd, args[0])
            if os.path.isfile(exe_in_cwd):
                executable = exe_in_cwd
        if not executable:
            executable = lit.util.which(args[0], cmd_shenv.env['PATH'])
        if not executable:
            raise InternalShellError(j, '%r: command not found' % args[0])

        # Replace uses of /dev/null with temporary files.
        if kAvoidDevNull:
            # In Python 2.x, basestring is the base class for all string (including unicode)
            # In Python 3.x, basestring no longer exist and str is always unicode
            try:
                str_type = basestring
            except NameError:
                str_type = str
            for i,arg in enumerate(args):
                if isinstance(arg, str_type) and kDevNull in arg:
                    f = tempfile.NamedTemporaryFile(delete=False)
                    f.close()
                    named_temp_files.append(f.name)
                    args[i] = arg.replace(kDevNull, f.name)

        # Expand all glob expressions
        args = expand_glob_expressions(args, cmd_shenv.cwd)

        # On Windows, do our own command line quoting for better compatibility
        # with some core utility distributions.
        if kIsWindows:
            args = quote_windows_command(args)

        try:
            procs.append(subprocess.Popen(args, cwd=cmd_shenv.cwd,
                                          executable = executable,
                                          stdin = stdin,
                                          stdout = stdout,
                                          stderr = stderr,
                                          env = cmd_shenv.env,
                                          close_fds = kUseCloseFDs))
            proc_not_counts.append(not_count)
            # Let the helper know about this process
            timeoutHelper.addProcess(procs[-1])
        except OSError as e:
            raise InternalShellError(j, 'Could not create process ({}) due to {}'.format(executable, e))

        # Immediately close stdin for any process taking stdin from us.
        if stdin == subprocess.PIPE:
            procs[-1].stdin.close()
            procs[-1].stdin = None

        # Update the current stdin source.
        if stdout == subprocess.PIPE:
            default_stdin = procs[-1].stdout
        elif stderrIsStdout:
            default_stdin = procs[-1].stderr
        else:
            default_stdin = subprocess.PIPE

    # Explicitly close any redirected files. We need to do this now because we
    # need to release any handles we may have on the temporary files (important
    # on Win32, for example). Since we have already spawned the subprocess, our
    # handles have already been transferred so we do not need them anymore.
    for (name, mode, f, path) in opened_files:
        f.close()

    # FIXME: There is probably still deadlock potential here. Yawn.
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

    # Read stderr out of the temp files.
    for i,f in stderrTempFiles:
        f.seek(0, 0)
        procData[i] = (procData[i][0], f.read())
        f.close()

    exitCode = None
    for i,(out,err) in enumerate(procData):
        res = procs[i].wait()
        # Detect Ctrl-C in subprocess.
        if res == -signal.SIGINT:
            raise KeyboardInterrupt
        if proc_not_counts[i] % 2:
            res = not res
        elif proc_not_counts[i] > 1:
            res = 1 if res != 0 else 0

        # Ensure the resulting output is always of string type.
        try:
            if out is None:
                out = ''
            else:
                out = to_string(out.decode('utf-8', errors='replace'))
        except:
            out = str(out)
        try:
            if err is None:
                err = ''
            else:
                err = to_string(err.decode('utf-8', errors='replace'))
        except:
            err = str(err)

        # Gather the redirected output files for failed commands.
        output_files = []
        if res != 0:
            for (name, mode, f, path) in sorted(opened_files):
                if path is not None and mode in ('w', 'a'):
                    try:
                        with open(path, 'rb') as f:
                            data = f.read()
                    except:
                        data = None
                    if data is not None:
                        output_files.append((name, path, data))

        results.append(ShellCommandResult(
            cmd.commands[i], out, err, res, timeoutHelper.timeoutReached(),
            output_files))
        if cmd.pipe_err:
            # Take the last failing exit code from the pipeline.
            if not exitCode or res != 0:
                exitCode = res
        else:
            exitCode = res

    # Remove any named temporary files we created.
    for f in named_temp_files:
        try:
            os.remove(f)
        except OSError:
            pass

    if cmd.negate:
        exitCode = not exitCode

    return exitCode

def executeScriptInternal(test, litConfig, tmpBase, commands, cwd):
    cmds = []
    for i, ln in enumerate(commands):
        ln = commands[i] = re.sub(kPdbgRegex, ": '\\1'; ", ln)
        try:
            cmds.append(ShUtil.ShParser(ln, litConfig.isWindows,
                                        test.config.pipefail).parse())
        except:
            return lit.Test.Result(Test.FAIL, "shell parser error on: %r" % ln)

    cmd = cmds[0]
    for c in cmds[1:]:
        cmd = ShUtil.Seq(cmd, '&&', c)

    results = []
    timeoutInfo = None
    try:
        shenv = ShellEnvironment(cwd, test.config.environment)
        exitCode, timeoutInfo = executeShCmd(cmd, shenv, results, timeout=litConfig.maxIndividualTestTime)
    except InternalShellError:
        e = sys.exc_info()[1]
        exitCode = 127
        results.append(
            ShellCommandResult(e.command, '', e.message, exitCode, False))

    out = err = ''
    for i,result in enumerate(results):
        # Write the command line run.
        out += '$ %s\n' % (' '.join('"%s"' % s
                                    for s in result.command.args),)

        # If nothing interesting happened, move on.
        if litConfig.maxIndividualTestTime == 0 and \
               result.exitCode == 0 and \
               not result.stdout.strip() and not result.stderr.strip():
            continue

        # Otherwise, something failed or was printed, show it.

        # Add the command output, if redirected.
        for (name, path, data) in result.outputFiles:
            if data.strip():
                out += "# redirected output from %r:\n" % (name,)
                data = to_string(data.decode('utf-8', errors='replace'))
                if len(data) > 1024:
                    out += data[:1024] + "\n...\n"
                    out += "note: data was truncated\n"
                else:
                    out += data
                out += "\n"

        if result.stdout.strip():
            out += '# command output:\n%s\n' % (result.stdout,)
        if result.stderr.strip():
            out += '# command stderr:\n%s\n' % (result.stderr,)
        if not result.stdout.strip() and not result.stderr.strip():
            out += "note: command had no output on stdout or stderr\n"

        # Show the error conditions:
        if result.exitCode != 0:
            # On Windows, a negative exit code indicates a signal, and those are
            # easier to recognize or look up if we print them in hex.
            if litConfig.isWindows and result.exitCode < 0:
                codeStr = hex(int(result.exitCode & 0xFFFFFFFF)).rstrip("L")
            else:
                codeStr = str(result.exitCode)
            out += "error: command failed with exit status: %s\n" % (
                codeStr,)
        if litConfig.maxIndividualTestTime > 0 and result.timeoutReached:
            out += 'error: command reached timeout: %s\n' % (
                str(result.timeoutReached),)

    return out, err, exitCode, timeoutInfo

def executeScript(test, litConfig, tmpBase, commands, cwd):
    bashPath = litConfig.getBashPath()
    isWin32CMDEXE = (litConfig.isWindows and not bashPath)
    script = tmpBase + '.script'
    if isWin32CMDEXE:
        script += '.bat'

    # Write script file
    mode = 'w'
    open_kwargs = {}
    if litConfig.isWindows and not isWin32CMDEXE:
        mode += 'b'  # Avoid CRLFs when writing bash scripts.
    elif sys.version_info > (3,0):
        open_kwargs['encoding'] = 'utf-8'
    f = open(script, mode, **open_kwargs)
    if isWin32CMDEXE:
        for i, ln in enumerate(commands):
            commands[i] = re.sub(kPdbgRegex, "echo '\\1' > nul && ", ln)
        if litConfig.echo_all_commands:
            f.write('@echo on\n')
        else:
            f.write('@echo off\n')
        f.write('\n@if %ERRORLEVEL% NEQ 0 EXIT\n'.join(commands))
    else:
        for i, ln in enumerate(commands):
            commands[i] = re.sub(kPdbgRegex, ": '\\1'; ", ln)
        if test.config.pipefail:
            f.write(b'set -o pipefail;' if mode == 'wb' else 'set -o pipefail;')
        if litConfig.echo_all_commands:
            f.write(b'set -x;' if mode == 'wb' else 'set -x;')
        if sys.version_info > (3,0) and mode == 'wb':
            f.write(bytes('{ ' + '; } &&\n{ '.join(commands) + '; }', 'utf-8'))
        else:
            f.write('{ ' + '; } &&\n{ '.join(commands) + '; }')
    f.write(b'\n' if mode == 'wb' else '\n')
    f.close()

    if isWin32CMDEXE:
        command = ['cmd','/c', script]
    else:
        if bashPath:
            command = [bashPath, script]
        else:
            command = ['/bin/sh', script]
        if litConfig.useValgrind:
            # FIXME: Running valgrind on sh is overkill. We probably could just
            # run on clang with no real loss.
            command = litConfig.valgrindArgs + command

    try:
        out, err, exitCode = lit.util.executeCommand(command, cwd=cwd,
                                       env=test.config.environment,
                                       timeout=litConfig.maxIndividualTestTime)
        return (out, err, exitCode, None)
    except lit.util.ExecuteCommandTimeoutException as e:
        return (e.out, e.err, e.exitCode, e.msg)

def parseIntegratedTestScriptCommands(source_path, keywords):
    """
    parseIntegratedTestScriptCommands(source_path) -> commands

    Parse the commands in an integrated test script file into a list of
    (line_number, command_type, line).
    """

    # This code is carefully written to be dual compatible with Python 2.5+ and
    # Python 3 without requiring input files to always have valid codings. The
    # trick we use is to open the file in binary mode and use the regular
    # expression library to find the commands, with it scanning strings in
    # Python2 and bytes in Python3.
    #
    # Once we find a match, we do require each script line to be decodable to
    # UTF-8, so we convert the outputs to UTF-8 before returning. This way the
    # remaining code can work with "strings" agnostic of the executing Python
    # version.

    keywords_re = re.compile(
        to_bytes("(%s)(.*)\n" % ("|".join(re.escape(k) for k in keywords),)))

    f = open(source_path, 'rb')
    try:
        # Read the entire file contents.
        data = f.read()

        # Ensure the data ends with a newline.
        if not data.endswith(to_bytes('\n')):
            data = data + to_bytes('\n')

        # Iterate over the matches.
        line_number = 1
        last_match_position = 0
        for match in keywords_re.finditer(data):
            # Compute the updated line number by counting the intervening
            # newlines.
            match_position = match.start()
            line_number += data.count(to_bytes('\n'), last_match_position,
                                      match_position)
            last_match_position = match_position

            # Convert the keyword and line to UTF-8 strings and yield the
            # command. Note that we take care to return regular strings in
            # Python 2, to avoid other code having to differentiate between the
            # str and unicode types.
            #
            # Opening the file in binary mode prevented Windows \r newline
            # characters from being converted to Unix \n newlines, so manually
            # strip those from the yielded lines.
            keyword,ln = match.groups()
            yield (line_number, to_string(keyword.decode('utf-8')),
                   to_string(ln.decode('utf-8').rstrip('\r')))
    finally:
        f.close()

def getTempPaths(test):
    """Get the temporary location, this is always relative to the test suite
    root, not test source root."""
    execpath = test.getExecPath()
    execdir,execbase = os.path.split(execpath)
    tmpDir = os.path.join(execdir, 'Output')
    tmpBase = os.path.join(tmpDir, execbase)
    return tmpDir, tmpBase

def colonNormalizePath(path):
    if kIsWindows:
        return re.sub(r'^(.):', r'\1', path.replace('\\', '/'))
    else:
        assert path[0] == '/'
        return path[1:]

def getDefaultSubstitutions(test, tmpDir, tmpBase, normalize_slashes=False):
    sourcepath = test.getSourcePath()
    sourcedir = os.path.dirname(sourcepath)

    # Normalize slashes, if requested.
    if normalize_slashes:
        sourcepath = sourcepath.replace('\\', '/')
        sourcedir = sourcedir.replace('\\', '/')
        tmpDir = tmpDir.replace('\\', '/')
        tmpBase = tmpBase.replace('\\', '/')

    substitutions = []
    substitutions.extend(test.config.substitutions)
    tmpName = tmpBase + '.tmp'
    baseName = os.path.basename(tmpBase)
    substitutions.extend([('%s', sourcepath),
                          ('%S', sourcedir),
                          ('%p', sourcedir),
                          ('%{pathsep}', os.pathsep),
                          ('%t', tmpName),
                          ('%basename_t', baseName),
                          ('%T', tmpDir)])

    # "%/[STpst]" should be normalized.
    substitutions.extend([
            ('%/s', sourcepath.replace('\\', '/')),
            ('%/S', sourcedir.replace('\\', '/')),
            ('%/p', sourcedir.replace('\\', '/')),
            ('%/t', tmpBase.replace('\\', '/') + '.tmp'),
            ('%/T', tmpDir.replace('\\', '/')),
            ])

    # "%{/[STpst]:regex_replacement}" should be normalized like "%/[STpst]" but we're
    # also in a regex replacement context of a s@@@ regex.
    def regex_escape(s):
        s = s.replace('@', r'\@')
        s = s.replace('&', r'\&')
        return s
    substitutions.extend([
            ('%{/s:regex_replacement}',
             regex_escape(sourcepath.replace('\\', '/'))),
            ('%{/S:regex_replacement}',
             regex_escape(sourcedir.replace('\\', '/'))),
            ('%{/p:regex_replacement}',
             regex_escape(sourcedir.replace('\\', '/'))),
            ('%{/t:regex_replacement}',
             regex_escape(tmpBase.replace('\\', '/')) + '.tmp'),
            ('%{/T:regex_replacement}',
             regex_escape(tmpDir.replace('\\', '/'))),
            ])

    # "%:[STpst]" are normalized paths without colons and without a leading
    # slash.
    substitutions.extend([
            ('%:s', colonNormalizePath(sourcepath)),
            ('%:S', colonNormalizePath(sourcedir)),
            ('%:p', colonNormalizePath(sourcedir)),
            ('%:t', colonNormalizePath(tmpBase + '.tmp')),
            ('%:T', colonNormalizePath(tmpDir)),
            ])
    return substitutions

def _memoize(f):
    cache = {}  # Intentionally unbounded, see applySubstitutions()
    def memoized(x):
        if x not in cache:
            cache[x] = f(x)
        return cache[x]
    return memoized

@_memoize
def _caching_re_compile(r):
    return re.compile(r)

def applySubstitutions(script, substitutions, recursion_limit=None):
    """
    Apply substitutions to the script.  Allow full regular expression syntax.
    Replace each matching occurrence of regular expression pattern a with
    substitution b in line ln.

    If a substitution expands into another substitution, it is expanded
    recursively until the line has no more expandable substitutions. If
    the line can still can be substituted after being substituted
    `recursion_limit` times, it is an error. If the `recursion_limit` is
    `None` (the default), no recursive substitution is performed at all.
    """

    # We use #_MARKER_# to hide %% while we do the other substitutions.
    def escape(ln):
        return _caching_re_compile('%%').sub('#_MARKER_#', ln)

    def unescape(ln):
        return _caching_re_compile('#_MARKER_#').sub('%', ln)

    def processLine(ln):
        # Apply substitutions
        for a,b in substitutions:
            if kIsWindows:
                b = b.replace("\\","\\\\")
            # re.compile() has a built-in LRU cache with 512 entries. In some
            # test suites lit ends up thrashing that cache, which made e.g.
            # check-llvm run 50% slower.  Use an explicit, unbounded cache
            # to prevent that from happening.  Since lit is fairly
            # short-lived, since the set of substitutions is fairly small, and
            # since thrashing has such bad consequences, not bounding the cache
            # seems reasonable.
            ln = _caching_re_compile(a).sub(str(b), escape(ln))

        # Strip the trailing newline and any extra whitespace.
        return ln.strip()

    def processLineToFixedPoint(ln):
        assert isinstance(recursion_limit, int) and recursion_limit >= 0
        origLine = ln
        steps = 0
        processed = processLine(ln)
        while processed != ln and steps < recursion_limit:
            ln = processed
            processed = processLine(ln)
            steps += 1

        if processed != ln:
            raise ValueError("Recursive substitution of '%s' did not complete "
                             "in the provided recursion limit (%s)" % \
                             (origLine, recursion_limit))

        return processed

    process = processLine if recursion_limit is None else processLineToFixedPoint
    
    return [unescape(process(ln)) for ln in script]


class ParserKind(object):
    """
    An enumeration representing the style of an integrated test keyword or
    command.

    TAG: A keyword taking no value. Ex 'END.'
    COMMAND: A keyword taking a list of shell commands. Ex 'RUN:'
    LIST: A keyword taking a comma-separated list of values.
    BOOLEAN_EXPR: A keyword taking a comma-separated list of
        boolean expressions. Ex 'XFAIL:'
    INTEGER: A keyword taking a single integer. Ex 'ALLOW_RETRIES:'
    CUSTOM: A keyword with custom parsing semantics.
    """
    TAG = 0
    COMMAND = 1
    LIST = 2
    BOOLEAN_EXPR = 3
    INTEGER = 4
    CUSTOM = 5

    @staticmethod
    def allowedKeywordSuffixes(value):
        return { ParserKind.TAG:          ['.'],
                 ParserKind.COMMAND:      [':'],
                 ParserKind.LIST:         [':'],
                 ParserKind.BOOLEAN_EXPR: [':'],
                 ParserKind.INTEGER:      [':'],
                 ParserKind.CUSTOM:       [':', '.']
               } [value]

    @staticmethod
    def str(value):
        return { ParserKind.TAG:          'TAG',
                 ParserKind.COMMAND:      'COMMAND',
                 ParserKind.LIST:         'LIST',
                 ParserKind.BOOLEAN_EXPR: 'BOOLEAN_EXPR',
                 ParserKind.INTEGER:      'INTEGER',
                 ParserKind.CUSTOM:       'CUSTOM'
               } [value]


class IntegratedTestKeywordParser(object):
    """A parser for LLVM/Clang style integrated test scripts.

    keyword: The keyword to parse for. It must end in either '.' or ':'.
    kind: An value of ParserKind.
    parser: A custom parser. This value may only be specified with
            ParserKind.CUSTOM.
    """
    def __init__(self, keyword, kind, parser=None, initial_value=None):
        allowedSuffixes = ParserKind.allowedKeywordSuffixes(kind)
        if len(keyword) == 0 or keyword[-1] not in allowedSuffixes:
            if len(allowedSuffixes) == 1:
                raise ValueError("Keyword '%s' of kind '%s' must end in '%s'"
                                 % (keyword, ParserKind.str(kind),
                                    allowedSuffixes[0]))
            else:
                raise ValueError("Keyword '%s' of kind '%s' must end in "
                                 " one of '%s'"
                                 % (keyword, ParserKind.str(kind),
                                    ' '.join(allowedSuffixes)))

        if parser is not None and kind != ParserKind.CUSTOM:
            raise ValueError("custom parsers can only be specified with "
                             "ParserKind.CUSTOM")
        self.keyword = keyword
        self.kind = kind
        self.parsed_lines = []
        self.value = initial_value
        self.parser = parser

        if kind == ParserKind.COMMAND:
            self.parser = lambda line_number, line, output: \
                                 self._handleCommand(line_number, line, output,
                                                     self.keyword)
        elif kind == ParserKind.LIST:
            self.parser = self._handleList
        elif kind == ParserKind.BOOLEAN_EXPR:
            self.parser = self._handleBooleanExpr
        elif kind == ParserKind.INTEGER:
            self.parser = self._handleSingleInteger
        elif kind == ParserKind.TAG:
            self.parser = self._handleTag
        elif kind == ParserKind.CUSTOM:
            if parser is None:
                raise ValueError("ParserKind.CUSTOM requires a custom parser")
            self.parser = parser
        else:
            raise ValueError("Unknown kind '%s'" % kind)

    def parseLine(self, line_number, line):
        try:
            self.parsed_lines += [(line_number, line)]
            self.value = self.parser(line_number, line, self.value)
        except ValueError as e:
            raise ValueError(str(e) + ("\nin %s directive on test line %d" %
                                       (self.keyword, line_number)))

    def getValue(self):
        return self.value

    @staticmethod
    def _handleTag(line_number, line, output):
        """A helper for parsing TAG type keywords"""
        return (not line.strip() or output)

    @staticmethod
    def _handleCommand(line_number, line, output, keyword):
        """A helper for parsing COMMAND type keywords"""
        # Trim trailing whitespace.
        line = line.rstrip()
        # Substitute line number expressions
        line = re.sub(r'%\(line\)', str(line_number), line)

        def replace_line_number(match):
            if match.group(1) == '+':
                return str(line_number + int(match.group(2)))
            if match.group(1) == '-':
                return str(line_number - int(match.group(2)))
        line = re.sub(r'%\(line *([\+-]) *(\d+)\)', replace_line_number, line)
        # Collapse lines with trailing '\\'.
        if output and output[-1][-1] == '\\':
            output[-1] = output[-1][:-1] + line
        else:
            if output is None:
                output = []
            pdbg = "%dbg({keyword} at line {line_number})".format(
                keyword=keyword,
                line_number=line_number)
            assert re.match(kPdbgRegex + "$", pdbg), \
                   "kPdbgRegex expected to match actual %dbg usage"
            line = "{pdbg} {real_command}".format(
                pdbg=pdbg,
                real_command=line)
            output.append(line)
        return output

    @staticmethod
    def _handleList(line_number, line, output):
        """A parser for LIST type keywords"""
        if output is None:
            output = []
        output.extend([s.strip() for s in line.split(',')])
        return output

    @staticmethod
    def _handleSingleInteger(line_number, line, output):
        """A parser for INTEGER type keywords"""
        if output is None:
            output = []
        try:
            n = int(line)
        except ValueError:
            raise ValueError("INTEGER parser requires the input to be an integer (got {})".format(line))
        output.append(n)
        return output

    @staticmethod
    def _handleBooleanExpr(line_number, line, output):
        """A parser for BOOLEAN_EXPR type keywords"""
        parts = [s.strip() for s in line.split(',') if s.strip() != '']
        if output and output[-1][-1] == '\\':
            output[-1] = output[-1][:-1] + parts[0]
            del parts[0]
        if output is None:
            output = []
        output.extend(parts)
        # Evaluate each expression to verify syntax.
        # We don't want any results, just the raised ValueError.
        for s in output:
            if s != '*' and not s.endswith('\\'):
                BooleanExpression.evaluate(s, [])
        return output


def _parseKeywords(sourcepath, additional_parsers=[],
                   require_script=True):
    """_parseKeywords

    Scan an LLVM/Clang style integrated test script and extract all the lines
    pertaining to a special parser. This includes 'RUN', 'XFAIL', 'REQUIRES',
    'UNSUPPORTED' and 'ALLOW_RETRIES', as well as other specified custom
    parsers.

    Returns a dictionary mapping each custom parser to its value after
    parsing the test.
    """
    # Install the built-in keyword parsers.
    script = []
    builtin_parsers = [
        IntegratedTestKeywordParser('RUN:', ParserKind.COMMAND, initial_value=script),
        IntegratedTestKeywordParser('XFAIL:', ParserKind.BOOLEAN_EXPR),
        IntegratedTestKeywordParser('REQUIRES:', ParserKind.BOOLEAN_EXPR),
        IntegratedTestKeywordParser('UNSUPPORTED:', ParserKind.BOOLEAN_EXPR),
        IntegratedTestKeywordParser('ALLOW_RETRIES:', ParserKind.INTEGER),
        IntegratedTestKeywordParser('END.', ParserKind.TAG)
    ]
    keyword_parsers = {p.keyword: p for p in builtin_parsers}

    # Install user-defined additional parsers.
    for parser in additional_parsers:
        if not isinstance(parser, IntegratedTestKeywordParser):
            raise ValueError('Additional parser must be an instance of '
                             'IntegratedTestKeywordParser')
        if parser.keyword in keyword_parsers:
            raise ValueError("Parser for keyword '%s' already exists"
                             % parser.keyword)
        keyword_parsers[parser.keyword] = parser

    # Collect the test lines from the script.
    for line_number, command_type, ln in \
            parseIntegratedTestScriptCommands(sourcepath,
                                              keyword_parsers.keys()):
        parser = keyword_parsers[command_type]
        parser.parseLine(line_number, ln)
        if command_type == 'END.' and parser.getValue() is True:
            break

    # Verify the script contains a run line.
    if require_script and not script:
        raise ValueError("Test has no 'RUN:' line")

    # Check for unterminated run lines.
    if script and script[-1][-1] == '\\':
        raise ValueError("Test has unterminated 'RUN:' lines (with '\\')")

    # Check boolean expressions for unterminated lines.
    for key in keyword_parsers:
        kp = keyword_parsers[key]
        if kp.kind != ParserKind.BOOLEAN_EXPR:
            continue
        value = kp.getValue()
        if value and value[-1][-1] == '\\':
            raise ValueError("Test has unterminated '{key}' lines (with '\\')"
                             .format(key=key))

    # Make sure there's at most one ALLOW_RETRIES: line
    allowed_retries = keyword_parsers['ALLOW_RETRIES:'].getValue()
    if allowed_retries and len(allowed_retries) > 1:
        raise ValueError("Test has more than one ALLOW_RETRIES lines")

    return {p.keyword: p.getValue() for p in keyword_parsers.values()}


def parseIntegratedTestScript(test, additional_parsers=[],
                              require_script=True):
    """parseIntegratedTestScript - Scan an LLVM/Clang style integrated test
    script and extract the lines to 'RUN' as well as 'XFAIL', 'REQUIRES',
    'UNSUPPORTED' and 'ALLOW_RETRIES' information into the given test.

    If additional parsers are specified then the test is also scanned for the
    keywords they specify and all matches are passed to the custom parser.

    If 'require_script' is False an empty script
    may be returned. This can be used for test formats where the actual script
    is optional or ignored.
    """
    # Parse the test sources and extract test properties
    try:
        parsed = _parseKeywords(test.getSourcePath(), additional_parsers,
                                require_script)
    except ValueError as e:
        return lit.Test.Result(Test.UNRESOLVED, str(e))
    script = parsed['RUN:'] or []
    test.xfails += parsed['XFAIL:'] or []
    test.requires += parsed['REQUIRES:'] or []
    test.unsupported += parsed['UNSUPPORTED:'] or []
    if parsed['ALLOW_RETRIES:']:
        test.allowed_retries = parsed['ALLOW_RETRIES:'][0]

    # Enforce REQUIRES:
    missing_required_features = test.getMissingRequiredFeatures()
    if missing_required_features:
        msg = ', '.join(missing_required_features)
        return lit.Test.Result(Test.UNSUPPORTED,
                               "Test requires the following unavailable "
                               "features: %s" % msg)

    # Enforce UNSUPPORTED:
    unsupported_features = test.getUnsupportedFeatures()
    if unsupported_features:
        msg = ', '.join(unsupported_features)
        return lit.Test.Result(
            Test.UNSUPPORTED,
            "Test does not support the following features "
            "and/or targets: %s" % msg)

    # Enforce limit_to_features.
    if not test.isWithinFeatureLimits():
        msg = ', '.join(test.config.limit_to_features)
        return lit.Test.Result(Test.UNSUPPORTED,
                               "Test does not require any of the features "
                               "specified in limit_to_features: %s" % msg)

    return script


def _runShTest(test, litConfig, useExternalSh, script, tmpBase):
    def runOnce(execdir):
        if useExternalSh:
            res = executeScript(test, litConfig, tmpBase, script, execdir)
        else:
            res = executeScriptInternal(test, litConfig, tmpBase, script, execdir)
        if isinstance(res, lit.Test.Result):
            return res

        out,err,exitCode,timeoutInfo = res
        if exitCode == 0:
            status = Test.PASS
        else:
            if timeoutInfo is None:
                status = Test.FAIL
            else:
                status = Test.TIMEOUT
        return out,err,exitCode,timeoutInfo,status

    # Create the output directory if it does not already exist.
    lit.util.mkdir_p(os.path.dirname(tmpBase))

    # Re-run failed tests up to test.allowed_retries times.
    execdir = os.path.dirname(test.getExecPath())
    attempts = test.allowed_retries + 1
    for i in range(attempts):
        res = runOnce(execdir)
        if isinstance(res, lit.Test.Result):
            return res

        out,err,exitCode,timeoutInfo,status = res
        if status != Test.FAIL:
            break

    # If we had to run the test more than once, count it as a flaky pass. These
    # will be printed separately in the test summary.
    if i > 0 and status == Test.PASS:
        status = Test.FLAKYPASS

    # Form the output log.
    output = """Script:\n--\n%s\n--\nExit Code: %d\n""" % (
        '\n'.join(script), exitCode)

    if timeoutInfo is not None:
        output += """Timeout: %s\n""" % (timeoutInfo,)
    output += "\n"

    # Append the outputs, if present.
    if out:
        output += """Command Output (stdout):\n--\n%s\n--\n""" % (out,)
    if err:
        output += """Command Output (stderr):\n--\n%s\n--\n""" % (err,)

    return lit.Test.Result(status, output)


def executeShTest(test, litConfig, useExternalSh,
                  extra_substitutions=[],
                  preamble_commands=[]):
    if test.config.unsupported:
        return lit.Test.Result(Test.UNSUPPORTED, 'Test is unsupported')

    script = list(preamble_commands)
    parsed = parseIntegratedTestScript(test, require_script=not script)
    if isinstance(parsed, lit.Test.Result):
        return parsed
    script += parsed

    if litConfig.noExecute:
        return lit.Test.Result(Test.PASS)

    tmpDir, tmpBase = getTempPaths(test)
    substitutions = list(extra_substitutions)
    substitutions += getDefaultSubstitutions(test, tmpDir, tmpBase,
                                             normalize_slashes=useExternalSh)
    script = applySubstitutions(script, substitutions,
                                recursion_limit=test.config.recursiveExpansionLimit)

    return _runShTest(test, litConfig, useExternalSh, script, tmpBase)
