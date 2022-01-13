"""
This LLDB module contains miscellaneous utilities.
Some of the test suite takes advantage of the utility functions defined here.
They can also be useful for general purpose lldb scripting.
"""

from __future__ import print_function
from __future__ import absolute_import

# System modules
import errno
import os
import re
import sys
import subprocess

# Third-party modules
from six import StringIO as SixStringIO
import six

# LLDB modules
import lldb
from . import lldbtest_config
from . import configuration

# How often failed simulator process launches are retried.
SIMULATOR_RETRY = 3

# ===================================================
# Utilities for locating/checking executable programs
# ===================================================

def is_exe(fpath):
    """Returns True if fpath is an executable."""
    return os.path.isfile(fpath) and os.access(fpath, os.X_OK)


def which(program):
    """Returns the full path to a program; None otherwise."""
    fpath, fname = os.path.split(program)
    if fpath:
        if is_exe(program):
            return program
    else:
        for path in os.environ["PATH"].split(os.pathsep):
            exe_file = os.path.join(path, program)
            if is_exe(exe_file):
                return exe_file
    return None

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    if not os.path.isdir(path):
        raise OSError(errno.ENOTDIR, "%s is not a directory"%path)


# ============================
# Dealing with SDK and triples
# ============================

def get_xcode_sdk(os, env):
    # Respect --apple-sdk <path> if it's specified. If the SDK is simply
    # mounted from some disk image, and not actually installed, this is the
    # only way to use it.
    if configuration.apple_sdk:
        return configuration.apple_sdk
    if os == "ios":
        if env == "simulator":
            return "iphonesimulator"
        if env == "macabi":
            return "macosx"
        return "iphoneos"
    elif os == "tvos":
        if env == "simulator":
            return "appletvsimulator"
        return "appletvos"
    elif os == "watchos":
        if env == "simulator":
            return "watchsimulator"
        return "watchos"
    return os


def get_xcode_sdk_version(sdk):
    return subprocess.check_output(
        ['xcrun', '--sdk', sdk, '--show-sdk-version']).rstrip().decode('utf-8')


def get_xcode_sdk_root(sdk):
    return subprocess.check_output(['xcrun', '--sdk', sdk, '--show-sdk-path'
                                    ]).rstrip().decode('utf-8')


def get_xcode_clang(sdk):
    return subprocess.check_output(['xcrun', '-sdk', sdk, '-f', 'clang'
                                    ]).rstrip().decode("utf-8")


# ===================================================
# Disassembly for an SBFunction or an SBSymbol object
# ===================================================


def disassemble(target, function_or_symbol):
    """Disassemble the function or symbol given a target.

    It returns the disassembly content in a string object.
    """
    buf = SixStringIO()
    insts = function_or_symbol.GetInstructions(target)
    for i in insts:
        print(i, file=buf)
    return buf.getvalue()

# ==========================================================
# Integer (byte size 1, 2, 4, and 8) to bytearray conversion
# ==========================================================


def int_to_bytearray(val, bytesize):
    """Utility function to convert an integer into a bytearray.

    It returns the bytearray in the little endian format.  It is easy to get the
    big endian format, just do ba.reverse() on the returned object.
    """
    import struct

    if bytesize == 1:
        return bytearray([val])

    # Little endian followed by a format character.
    template = "<%c"
    if bytesize == 2:
        fmt = template % 'h'
    elif bytesize == 4:
        fmt = template % 'i'
    elif bytesize == 4:
        fmt = template % 'q'
    else:
        return None

    packed = struct.pack(fmt, val)
    return bytearray(packed)


def bytearray_to_int(bytes, bytesize):
    """Utility function to convert a bytearray into an integer.

    It interprets the bytearray in the little endian format. For a big endian
    bytearray, just do ba.reverse() on the object before passing it in.
    """
    import struct

    if bytesize == 1:
        return bytes[0]

    # Little endian followed by a format character.
    template = "<%c"
    if bytesize == 2:
        fmt = template % 'h'
    elif bytesize == 4:
        fmt = template % 'i'
    elif bytesize == 4:
        fmt = template % 'q'
    else:
        return None

    unpacked = struct.unpack_from(fmt, bytes)
    return unpacked[0]


# ==============================================================
# Get the description of an lldb object or None if not available
# ==============================================================
def get_description(obj, option=None):
    """Calls lldb_obj.GetDescription() and returns a string, or None.

    For SBTarget, SBBreakpointLocation, and SBWatchpoint lldb objects, an extra
    option can be passed in to describe the detailed level of description
    desired:
        o lldb.eDescriptionLevelBrief
        o lldb.eDescriptionLevelFull
        o lldb.eDescriptionLevelVerbose
    """
    method = getattr(obj, 'GetDescription')
    if not method:
        return None
    tuple = (lldb.SBTarget, lldb.SBBreakpointLocation, lldb.SBWatchpoint)
    if isinstance(obj, tuple):
        if option is None:
            option = lldb.eDescriptionLevelBrief

    stream = lldb.SBStream()
    if option is None:
        success = method(stream)
    else:
        success = method(stream, option)
    if not success:
        return None
    return stream.GetData()


# =================================================
# Convert some enum value to its string counterpart
# =================================================

def state_type_to_str(enum):
    """Returns the stateType string given an enum."""
    if enum == lldb.eStateInvalid:
        return "invalid"
    elif enum == lldb.eStateUnloaded:
        return "unloaded"
    elif enum == lldb.eStateConnected:
        return "connected"
    elif enum == lldb.eStateAttaching:
        return "attaching"
    elif enum == lldb.eStateLaunching:
        return "launching"
    elif enum == lldb.eStateStopped:
        return "stopped"
    elif enum == lldb.eStateRunning:
        return "running"
    elif enum == lldb.eStateStepping:
        return "stepping"
    elif enum == lldb.eStateCrashed:
        return "crashed"
    elif enum == lldb.eStateDetached:
        return "detached"
    elif enum == lldb.eStateExited:
        return "exited"
    elif enum == lldb.eStateSuspended:
        return "suspended"
    else:
        raise Exception("Unknown StateType enum")


def stop_reason_to_str(enum):
    """Returns the stopReason string given an enum."""
    if enum == lldb.eStopReasonInvalid:
        return "invalid"
    elif enum == lldb.eStopReasonNone:
        return "none"
    elif enum == lldb.eStopReasonTrace:
        return "trace"
    elif enum == lldb.eStopReasonBreakpoint:
        return "breakpoint"
    elif enum == lldb.eStopReasonWatchpoint:
        return "watchpoint"
    elif enum == lldb.eStopReasonExec:
        return "exec"
    elif enum == lldb.eStopReasonFork:
        return "fork"
    elif enum == lldb.eStopReasonVFork:
        return "vfork"
    elif enum == lldb.eStopReasonVForkDone:
        return "vforkdone"
    elif enum == lldb.eStopReasonSignal:
        return "signal"
    elif enum == lldb.eStopReasonException:
        return "exception"
    elif enum == lldb.eStopReasonPlanComplete:
        return "plancomplete"
    elif enum == lldb.eStopReasonThreadExiting:
        return "threadexiting"
    else:
        raise Exception("Unknown StopReason enum")


def symbol_type_to_str(enum):
    """Returns the symbolType string given an enum."""
    if enum == lldb.eSymbolTypeInvalid:
        return "invalid"
    elif enum == lldb.eSymbolTypeAbsolute:
        return "absolute"
    elif enum == lldb.eSymbolTypeCode:
        return "code"
    elif enum == lldb.eSymbolTypeData:
        return "data"
    elif enum == lldb.eSymbolTypeTrampoline:
        return "trampoline"
    elif enum == lldb.eSymbolTypeRuntime:
        return "runtime"
    elif enum == lldb.eSymbolTypeException:
        return "exception"
    elif enum == lldb.eSymbolTypeSourceFile:
        return "sourcefile"
    elif enum == lldb.eSymbolTypeHeaderFile:
        return "headerfile"
    elif enum == lldb.eSymbolTypeObjectFile:
        return "objectfile"
    elif enum == lldb.eSymbolTypeCommonBlock:
        return "commonblock"
    elif enum == lldb.eSymbolTypeBlock:
        return "block"
    elif enum == lldb.eSymbolTypeLocal:
        return "local"
    elif enum == lldb.eSymbolTypeParam:
        return "param"
    elif enum == lldb.eSymbolTypeVariable:
        return "variable"
    elif enum == lldb.eSymbolTypeVariableType:
        return "variabletype"
    elif enum == lldb.eSymbolTypeLineEntry:
        return "lineentry"
    elif enum == lldb.eSymbolTypeLineHeader:
        return "lineheader"
    elif enum == lldb.eSymbolTypeScopeBegin:
        return "scopebegin"
    elif enum == lldb.eSymbolTypeScopeEnd:
        return "scopeend"
    elif enum == lldb.eSymbolTypeAdditional:
        return "additional"
    elif enum == lldb.eSymbolTypeCompiler:
        return "compiler"
    elif enum == lldb.eSymbolTypeInstrumentation:
        return "instrumentation"
    elif enum == lldb.eSymbolTypeUndefined:
        return "undefined"


def value_type_to_str(enum):
    """Returns the valueType string given an enum."""
    if enum == lldb.eValueTypeInvalid:
        return "invalid"
    elif enum == lldb.eValueTypeVariableGlobal:
        return "global_variable"
    elif enum == lldb.eValueTypeVariableStatic:
        return "static_variable"
    elif enum == lldb.eValueTypeVariableArgument:
        return "argument_variable"
    elif enum == lldb.eValueTypeVariableLocal:
        return "local_variable"
    elif enum == lldb.eValueTypeRegister:
        return "register"
    elif enum == lldb.eValueTypeRegisterSet:
        return "register_set"
    elif enum == lldb.eValueTypeConstResult:
        return "constant_result"
    else:
        raise Exception("Unknown ValueType enum")


# ==================================================
# Get stopped threads due to each stop reason.
# ==================================================

def sort_stopped_threads(process,
                         breakpoint_threads=None,
                         crashed_threads=None,
                         watchpoint_threads=None,
                         signal_threads=None,
                         exiting_threads=None,
                         other_threads=None):
    """ Fills array *_threads with threads stopped for the corresponding stop
        reason.
    """
    for lst in [breakpoint_threads,
                watchpoint_threads,
                signal_threads,
                exiting_threads,
                other_threads]:
        if lst is not None:
            lst[:] = []

    for thread in process:
        dispatched = False
        for (reason, list) in [(lldb.eStopReasonBreakpoint, breakpoint_threads),
                               (lldb.eStopReasonException, crashed_threads),
                               (lldb.eStopReasonWatchpoint, watchpoint_threads),
                               (lldb.eStopReasonSignal, signal_threads),
                               (lldb.eStopReasonThreadExiting, exiting_threads),
                               (None, other_threads)]:
            if not dispatched and list is not None:
                if thread.GetStopReason() == reason or reason is None:
                    list.append(thread)
                    dispatched = True

# ==================================================
# Utility functions for setting breakpoints
# ==================================================

def run_break_set_by_script(
        test,
        class_name,
        extra_options=None,
        num_expected_locations=1):
    """Set a scripted breakpoint.  Check that it got the right number of locations."""
    test.assertTrue(class_name is not None, "Must pass in a class name.")
    command = "breakpoint set -P " + class_name
    if extra_options is not None:
        command += " " + extra_options

    break_results = run_break_set_command(test, command)
    check_breakpoint_result(test, break_results, num_locations=num_expected_locations)
    return get_bpno_from_match(break_results)

def run_break_set_by_file_and_line(
        test,
        file_name,
        line_number,
        extra_options=None,
        num_expected_locations=1,
        loc_exact=False,
        module_name=None):
    """Set a breakpoint by file and line, returning the breakpoint number.

    If extra_options is not None, then we append it to the breakpoint set command.

    If num_expected_locations is -1, we check that we got AT LEAST one location. If num_expected_locations is -2, we don't
    check the actual number at all. Otherwise, we check that num_expected_locations equals the number of locations.

    If loc_exact is true, we check that there is one location, and that location must be at the input file and line number."""

    if file_name is None:
        command = 'breakpoint set -l %d' % (line_number)
    else:
        command = 'breakpoint set -f "%s" -l %d' % (file_name, line_number)

    if module_name:
        command += " --shlib '%s'" % (module_name)

    if extra_options:
        command += " " + extra_options

    break_results = run_break_set_command(test, command)

    if num_expected_locations == 1 and loc_exact:
        check_breakpoint_result(
            test,
            break_results,
            num_locations=num_expected_locations,
            file_name=file_name,
            line_number=line_number,
            module_name=module_name)
    else:
        check_breakpoint_result(
            test,
            break_results,
            num_locations=num_expected_locations)

    return get_bpno_from_match(break_results)


def run_break_set_by_symbol(
        test,
        symbol,
        extra_options=None,
        num_expected_locations=-1,
        sym_exact=False,
        module_name=None):
    """Set a breakpoint by symbol name.  Common options are the same as run_break_set_by_file_and_line.

    If sym_exact is true, then the output symbol must match the input exactly, otherwise we do a substring match."""
    command = 'breakpoint set -n "%s"' % (symbol)

    if module_name:
        command += " --shlib '%s'" % (module_name)

    if extra_options:
        command += " " + extra_options

    break_results = run_break_set_command(test, command)

    if num_expected_locations == 1 and sym_exact:
        check_breakpoint_result(
            test,
            break_results,
            num_locations=num_expected_locations,
            symbol_name=symbol,
            module_name=module_name)
    else:
        check_breakpoint_result(
            test,
            break_results,
            num_locations=num_expected_locations)

    return get_bpno_from_match(break_results)


def run_break_set_by_selector(
        test,
        selector,
        extra_options=None,
        num_expected_locations=-1,
        module_name=None):
    """Set a breakpoint by selector.  Common options are the same as run_break_set_by_file_and_line."""

    command = 'breakpoint set -S "%s"' % (selector)

    if module_name:
        command += ' --shlib "%s"' % (module_name)

    if extra_options:
        command += " " + extra_options

    break_results = run_break_set_command(test, command)

    if num_expected_locations == 1:
        check_breakpoint_result(
            test,
            break_results,
            num_locations=num_expected_locations,
            symbol_name=selector,
            symbol_match_exact=False,
            module_name=module_name)
    else:
        check_breakpoint_result(
            test,
            break_results,
            num_locations=num_expected_locations)

    return get_bpno_from_match(break_results)


def run_break_set_by_regexp(
        test,
        regexp,
        extra_options=None,
        num_expected_locations=-1):
    """Set a breakpoint by regular expression match on symbol name.  Common options are the same as run_break_set_by_file_and_line."""

    command = 'breakpoint set -r "%s"' % (regexp)
    if extra_options:
        command += " " + extra_options

    break_results = run_break_set_command(test, command)

    check_breakpoint_result(
        test,
        break_results,
        num_locations=num_expected_locations)

    return get_bpno_from_match(break_results)


def run_break_set_by_source_regexp(
        test,
        regexp,
        extra_options=None,
        num_expected_locations=-1):
    """Set a breakpoint by source regular expression.  Common options are the same as run_break_set_by_file_and_line."""
    command = 'breakpoint set -p "%s"' % (regexp)
    if extra_options:
        command += " " + extra_options

    break_results = run_break_set_command(test, command)

    check_breakpoint_result(
        test,
        break_results,
        num_locations=num_expected_locations)

    return get_bpno_from_match(break_results)

def run_break_set_by_file_colon_line(
        test,
        specifier,
        path,
        line_number,
        column_number = 0,
        extra_options=None,
        num_expected_locations=-1):
    command = 'breakpoint set -y "%s"'%(specifier)
    if extra_options:
        command += " " + extra_options

    print("About to run: '%s'", command)
    break_results = run_break_set_command(test, command)
    check_breakpoint_result(
        test,
        break_results,
        num_locations = num_expected_locations,
        file_name = path,
        line_number = line_number,
        column_number = column_number)

    return get_bpno_from_match(break_results)

def run_break_set_command(test, command):
    """Run the command passed in - it must be some break set variant - and analyze the result.
    Returns a dictionary of information gleaned from the command-line results.
    Will assert if the breakpoint setting fails altogether.

    Dictionary will contain:
        bpno          - breakpoint of the newly created breakpoint, -1 on error.
        num_locations - number of locations set for the breakpoint.

    If there is only one location, the dictionary MAY contain:
        file          - source file name
        line_no       - source line number
        column        - source column number
        symbol        - symbol name
        inline_symbol - inlined symbol name
        offset        - offset from the original symbol
        module        - module
        address       - address at which the breakpoint was set."""

    patterns = [
        r"^Breakpoint (?P<bpno>[0-9]+): (?P<num_locations>[0-9]+) locations\.$",
        r"^Breakpoint (?P<bpno>[0-9]+): (?P<num_locations>no) locations \(pending\)\.",
        r"^Breakpoint (?P<bpno>[0-9]+): where = (?P<module>.*)`(?P<symbol>[+\-]{0,1}[^+]+)( \+ (?P<offset>[0-9]+)){0,1}( \[inlined\] (?P<inline_symbol>.*)){0,1} at (?P<file>[^:]+):(?P<line_no>[0-9]+)(?P<column>(:[0-9]+)?), address = (?P<address>0x[0-9a-fA-F]+)$",
        r"^Breakpoint (?P<bpno>[0-9]+): where = (?P<module>.*)`(?P<symbol>.*)( \+ (?P<offset>[0-9]+)){0,1}, address = (?P<address>0x[0-9a-fA-F]+)$"]
    match_object = test.match(command, patterns)
    break_results = match_object.groupdict()

    # We always insert the breakpoint number, setting it to -1 if we couldn't find it
    # Also, make sure it gets stored as an integer.
    if not 'bpno' in break_results:
        break_results['bpno'] = -1
    else:
        break_results['bpno'] = int(break_results['bpno'])

    # We always insert the number of locations
    # If ONE location is set for the breakpoint, then the output doesn't mention locations, but it has to be 1...
    # We also make sure it is an integer.

    if not 'num_locations' in break_results:
        num_locations = 1
    else:
        num_locations = break_results['num_locations']
        if num_locations == 'no':
            num_locations = 0
        else:
            num_locations = int(break_results['num_locations'])

    break_results['num_locations'] = num_locations

    if 'line_no' in break_results:
        break_results['line_no'] = int(break_results['line_no'])

    return break_results


def get_bpno_from_match(break_results):
    return int(break_results['bpno'])


def check_breakpoint_result(
        test,
        break_results,
        file_name=None,
        line_number=-1,
        column_number=0,
        symbol_name=None,
        symbol_match_exact=True,
        module_name=None,
        offset=-1,
        num_locations=-1):

    out_num_locations = break_results['num_locations']

    if num_locations == -1:
        test.assertTrue(out_num_locations > 0,
                        "Expecting one or more locations, got none.")
    elif num_locations != -2:
        test.assertTrue(
            num_locations == out_num_locations,
            "Expecting %d locations, got %d." %
            (num_locations,
             out_num_locations))

    if file_name:
        out_file_name = ""
        if 'file' in break_results:
            out_file_name = break_results['file']
        test.assertTrue(
            file_name.endswith(out_file_name),
            "Breakpoint file name '%s' doesn't match resultant name '%s'." %
            (file_name,
             out_file_name))

    if line_number != -1:
        out_line_number = -1
        if 'line_no' in break_results:
            out_line_number = break_results['line_no']

        test.assertTrue(
            line_number == out_line_number,
            "Breakpoint line number %s doesn't match resultant line %s." %
            (line_number,
             out_line_number))

    if column_number != 0:
        out_column_number = 0
        if 'column' in break_results:
            out_column_number = break_results['column']

        test.assertTrue(
            column_number == out_column_number,
            "Breakpoint column number %s doesn't match resultant column %s." %
            (column_number,
             out_column_number))

    if symbol_name:
        out_symbol_name = ""
        # Look first for the inlined symbol name, otherwise use the symbol
        # name:
        if 'inline_symbol' in break_results and break_results['inline_symbol']:
            out_symbol_name = break_results['inline_symbol']
        elif 'symbol' in break_results:
            out_symbol_name = break_results['symbol']

        if symbol_match_exact:
            test.assertTrue(
                symbol_name == out_symbol_name,
                "Symbol name '%s' doesn't match resultant symbol '%s'." %
                (symbol_name,
                 out_symbol_name))
        else:
            test.assertTrue(
                out_symbol_name.find(symbol_name) != -
                1,
                "Symbol name '%s' isn't in resultant symbol '%s'." %
                (symbol_name,
                 out_symbol_name))

    if module_name:
        out_module_name = None
        if 'module' in break_results:
            out_module_name = break_results['module']

        test.assertTrue(
            module_name.find(out_module_name) != -
            1,
            "Symbol module name '%s' isn't in expected module name '%s'." %
            (out_module_name,
             module_name))

# ==================================================
# Utility functions related to Threads and Processes
# ==================================================


def get_stopped_threads(process, reason):
    """Returns the thread(s) with the specified stop reason in a list.

    The list can be empty if no such thread exists.
    """
    threads = []
    for t in process:
        if t.GetStopReason() == reason:
            threads.append(t)
    return threads


def get_stopped_thread(process, reason):
    """A convenience function which returns the first thread with the given stop
    reason or None.

    Example usages:

    1. Get the stopped thread due to a breakpoint condition

    ...
        from lldbutil import get_stopped_thread
        thread = get_stopped_thread(process, lldb.eStopReasonPlanComplete)
        self.assertTrue(thread.IsValid(), "There should be a thread stopped due to breakpoint condition")
    ...

    2. Get the thread stopped due to a breakpoint

    ...
        from lldbutil import get_stopped_thread
        thread = get_stopped_thread(process, lldb.eStopReasonBreakpoint)
        self.assertTrue(thread.IsValid(), "There should be a thread stopped due to breakpoint")
    ...

    """
    threads = get_stopped_threads(process, reason)
    if len(threads) == 0:
        return None
    return threads[0]


def get_threads_stopped_at_breakpoint_id(process, bpid):
    """ For a stopped process returns the thread stopped at the breakpoint passed in bkpt"""
    stopped_threads = []
    threads = []

    stopped_threads = get_stopped_threads(process, lldb.eStopReasonBreakpoint)

    if len(stopped_threads) == 0:
        return threads

    for thread in stopped_threads:
        # Make sure we've hit our breakpoint...
        break_id = thread.GetStopReasonDataAtIndex(0)
        if break_id == bpid:
            threads.append(thread)

    return threads


def get_threads_stopped_at_breakpoint(process, bkpt):
    return get_threads_stopped_at_breakpoint_id(process, bkpt.GetID())


def get_one_thread_stopped_at_breakpoint_id(
        process, bpid, require_exactly_one=True):
    threads = get_threads_stopped_at_breakpoint_id(process, bpid)
    if len(threads) == 0:
        return None
    if require_exactly_one and len(threads) != 1:
        return None

    return threads[0]


def get_one_thread_stopped_at_breakpoint(
        process, bkpt, require_exactly_one=True):
    return get_one_thread_stopped_at_breakpoint_id(
        process, bkpt.GetID(), require_exactly_one)


def is_thread_crashed(test, thread):
    """In the test suite we dereference a null pointer to simulate a crash. The way this is
    reported depends on the platform."""
    if test.platformIsDarwin():
        return thread.GetStopReason(
        ) == lldb.eStopReasonException and "EXC_BAD_ACCESS" in thread.GetStopDescription(100)
    elif test.getPlatform() == "linux":
        return thread.GetStopReason() == lldb.eStopReasonSignal and thread.GetStopReasonDataAtIndex(
            0) == thread.GetProcess().GetUnixSignals().GetSignalNumberFromName("SIGSEGV")
    elif test.getPlatform() == "windows":
        return "Exception 0xc0000005" in thread.GetStopDescription(200)
    else:
        return "invalid address" in thread.GetStopDescription(100)


def get_crashed_threads(test, process):
    threads = []
    if process.GetState() != lldb.eStateStopped:
        return threads
    for thread in process:
        if is_thread_crashed(test, thread):
            threads.append(thread)
    return threads

# Helper functions for run_to_{source,name}_breakpoint:

def run_to_breakpoint_make_target(test, exe_name = "a.out", in_cwd = True):
    if in_cwd:
        exe = test.getBuildArtifact(exe_name)

    # Create the target
    target = test.dbg.CreateTarget(exe)
    test.assertTrue(target, "Target: %s is not valid."%(exe_name))

    # Set environment variables for the inferior.
    if lldbtest_config.inferior_env:
        test.runCmd('settings set target.env-vars {}'.format(
            lldbtest_config.inferior_env))

    return target

def run_to_breakpoint_do_run(test, target, bkpt, launch_info = None,
                             only_one_thread = True, extra_images = None):

    # Launch the process, and do not stop at the entry point.
    if not launch_info:
        launch_info = target.GetLaunchInfo()
        launch_info.SetWorkingDirectory(test.get_process_working_directory())

    if extra_images:
        environ = test.registerSharedLibrariesWithTarget(target, extra_images)
        launch_info.SetEnvironmentEntries(environ, True)

    error = lldb.SBError()
    process = target.Launch(launch_info, error)

    # Unfortunate workaround for the iPhone simulator.
    retry = SIMULATOR_RETRY
    while (retry and error.Fail() and error.GetCString() and
           "Unable to boot the Simulator" in error.GetCString()):
        retry -= 1
        print("** Simulator is unresponsive. Retrying %d more time(s)"%retry)
        import time
        time.sleep(60)
        error = lldb.SBError()
        process = target.Launch(launch_info, error)

    test.assertTrue(process,
                    "Could not create a valid process for %s: %s" %
                    (target.GetExecutable().GetFilename(), error.GetCString()))
    test.assertFalse(error.Fail(),
                     "Process launch failed: %s" % (error.GetCString()))

    test.assertEqual(process.GetState(), lldb.eStateStopped)

    # Frame #0 should be at our breakpoint.
    threads = get_threads_stopped_at_breakpoint(
                process, bkpt)

    num_threads = len(threads)
    if only_one_thread:
        test.assertEqual(num_threads, 1, "Expected 1 thread to stop at breakpoint, %d did."%(num_threads))
    else:
        test.assertGreater(num_threads, 0, "No threads stopped at breakpoint")

    thread = threads[0]
    return (target, process, thread, bkpt)

def run_to_name_breakpoint (test, bkpt_name, launch_info = None,
                            exe_name = "a.out",
                            bkpt_module = None,
                            in_cwd = True,
                            only_one_thread = True,
                            extra_images = None):
    """Start up a target, using exe_name as the executable, and run it to
       a breakpoint set by name on bkpt_name restricted to bkpt_module.

       If you want to pass in launch arguments or environment
       variables, you can optionally pass in an SBLaunchInfo.  If you
       do that, remember to set the working directory as well.

       If your executable isn't called a.out, you can pass that in.
       And if your executable isn't in the CWD, pass in the absolute
       path to the executable in exe_name, and set in_cwd to False.

       If you need to restrict the breakpoint to a particular module,
       pass the module name (a string not a FileSpec) in bkpt_module.  If
       nothing is passed in setting will be unrestricted.

       If the target isn't valid, the breakpoint isn't found, or hit, the
       function will cause a testsuite failure.

       If successful it returns a tuple with the target process and
       thread that hit the breakpoint, and the breakpoint that we set
       for you.

       If only_one_thread is true, we require that there be only one
       thread stopped at the breakpoint.  Otherwise we only require one
       or more threads stop there.  If there are more than one, we return
       the first thread that stopped.
    """

    target = run_to_breakpoint_make_target(test, exe_name, in_cwd)

    breakpoint = target.BreakpointCreateByName(bkpt_name, bkpt_module)


    test.assertTrue(breakpoint.GetNumLocations() > 0,
                    "No locations found for name breakpoint: '%s'."%(bkpt_name))
    return run_to_breakpoint_do_run(test, target, breakpoint, launch_info,
                                    only_one_thread, extra_images)

def run_to_source_breakpoint(test, bkpt_pattern, source_spec,
                             launch_info = None, exe_name = "a.out",
                             bkpt_module = None,
                             in_cwd = True,
                             only_one_thread = True,
                             extra_images = None):
    """Start up a target, using exe_name as the executable, and run it to
       a breakpoint set by source regex bkpt_pattern.

       The rest of the behavior is the same as run_to_name_breakpoint.
    """

    target = run_to_breakpoint_make_target(test, exe_name, in_cwd)
    # Set the breakpoints
    breakpoint = target.BreakpointCreateBySourceRegex(
            bkpt_pattern, source_spec, bkpt_module)
    test.assertTrue(breakpoint.GetNumLocations() > 0,
        'No locations found for source breakpoint: "%s", file: "%s", dir: "%s"'
        %(bkpt_pattern, source_spec.GetFilename(), source_spec.GetDirectory()))
    return run_to_breakpoint_do_run(test, target, breakpoint, launch_info,
                                    only_one_thread, extra_images)

def run_to_line_breakpoint(test, source_spec, line_number, column = 0,
                           launch_info = None, exe_name = "a.out",
                           bkpt_module = None,
                           in_cwd = True,
                           only_one_thread = True,
                           extra_images = None):
    """Start up a target, using exe_name as the executable, and run it to
       a breakpoint set by (source_spec, line_number(, column)).

       The rest of the behavior is the same as run_to_name_breakpoint.
    """

    target = run_to_breakpoint_make_target(test, exe_name, in_cwd)
    # Set the breakpoints
    breakpoint = target.BreakpointCreateByLocation(
        source_spec, line_number, column, 0, lldb.SBFileSpecList())
    test.assertTrue(breakpoint.GetNumLocations() > 0,
        'No locations found for line breakpoint: "%s:%d(:%d)", dir: "%s"'
        %(source_spec.GetFilename(), line_number, column,
          source_spec.GetDirectory()))
    return run_to_breakpoint_do_run(test, target, breakpoint, launch_info,
                                    only_one_thread, extra_images)


def continue_to_breakpoint(process, bkpt):
    """ Continues the process, if it stops, returns the threads stopped at bkpt; otherwise, returns None"""
    process.Continue()
    if process.GetState() != lldb.eStateStopped:
        return None
    else:
        return get_threads_stopped_at_breakpoint(process, bkpt)


def get_caller_symbol(thread):
    """
    Returns the symbol name for the call site of the leaf function.
    """
    depth = thread.GetNumFrames()
    if depth <= 1:
        return None
    caller = thread.GetFrameAtIndex(1).GetSymbol()
    if caller:
        return caller.GetName()
    else:
        return None


def get_function_names(thread):
    """
    Returns a sequence of function names from the stack frames of this thread.
    """
    def GetFuncName(i):
        return thread.GetFrameAtIndex(i).GetFunctionName()

    return list(map(GetFuncName, list(range(thread.GetNumFrames()))))


def get_symbol_names(thread):
    """
    Returns a sequence of symbols for this thread.
    """
    def GetSymbol(i):
        return thread.GetFrameAtIndex(i).GetSymbol().GetName()

    return list(map(GetSymbol, list(range(thread.GetNumFrames()))))


def get_pc_addresses(thread):
    """
    Returns a sequence of pc addresses for this thread.
    """
    def GetPCAddress(i):
        return thread.GetFrameAtIndex(i).GetPCAddress()

    return list(map(GetPCAddress, list(range(thread.GetNumFrames()))))


def get_filenames(thread):
    """
    Returns a sequence of file names from the stack frames of this thread.
    """
    def GetFilename(i):
        return thread.GetFrameAtIndex(
            i).GetLineEntry().GetFileSpec().GetFilename()

    return list(map(GetFilename, list(range(thread.GetNumFrames()))))


def get_line_numbers(thread):
    """
    Returns a sequence of line numbers from the stack frames of this thread.
    """
    def GetLineNumber(i):
        return thread.GetFrameAtIndex(i).GetLineEntry().GetLine()

    return list(map(GetLineNumber, list(range(thread.GetNumFrames()))))


def get_module_names(thread):
    """
    Returns a sequence of module names from the stack frames of this thread.
    """
    def GetModuleName(i):
        return thread.GetFrameAtIndex(
            i).GetModule().GetFileSpec().GetFilename()

    return list(map(GetModuleName, list(range(thread.GetNumFrames()))))


def get_stack_frames(thread):
    """
    Returns a sequence of stack frames for this thread.
    """
    def GetStackFrame(i):
        return thread.GetFrameAtIndex(i)

    return list(map(GetStackFrame, list(range(thread.GetNumFrames()))))


def print_stacktrace(thread, string_buffer=False):
    """Prints a simple stack trace of this thread."""

    output = SixStringIO() if string_buffer else sys.stdout
    target = thread.GetProcess().GetTarget()

    depth = thread.GetNumFrames()

    mods = get_module_names(thread)
    funcs = get_function_names(thread)
    symbols = get_symbol_names(thread)
    files = get_filenames(thread)
    lines = get_line_numbers(thread)
    addrs = get_pc_addresses(thread)

    if thread.GetStopReason() != lldb.eStopReasonInvalid:
        desc = "stop reason=" + stop_reason_to_str(thread.GetStopReason())
    else:
        desc = ""
    print(
        "Stack trace for thread id={0:#x} name={1} queue={2} ".format(
            thread.GetThreadID(),
            thread.GetName(),
            thread.GetQueueName()) + desc,
        file=output)

    for i in range(depth):
        frame = thread.GetFrameAtIndex(i)
        function = frame.GetFunction()

        load_addr = addrs[i].GetLoadAddress(target)
        if not function:
            file_addr = addrs[i].GetFileAddress()
            start_addr = frame.GetSymbol().GetStartAddress().GetFileAddress()
            symbol_offset = file_addr - start_addr
            print(
                "  frame #{num}: {addr:#016x} {mod}`{symbol} + {offset}".format(
                    num=i,
                    addr=load_addr,
                    mod=mods[i],
                    symbol=symbols[i],
                    offset=symbol_offset),
                file=output)
        else:
            print(
                "  frame #{num}: {addr:#016x} {mod}`{func} at {file}:{line} {args}".format(
                    num=i,
                    addr=load_addr,
                    mod=mods[i],
                    func='%s [inlined]' %
                    funcs[i] if frame.IsInlined() else funcs[i],
                    file=files[i],
                    line=lines[i],
                    args=get_args_as_string(
                        frame,
                        showFuncName=False) if not frame.IsInlined() else '()'),
                file=output)

    if string_buffer:
        return output.getvalue()


def print_stacktraces(process, string_buffer=False):
    """Prints the stack traces of all the threads."""

    output = SixStringIO() if string_buffer else sys.stdout

    print("Stack traces for " + str(process), file=output)

    for thread in process:
        print(print_stacktrace(thread, string_buffer=True), file=output)

    if string_buffer:
        return output.getvalue()


def expect_state_changes(test, listener, process, states, timeout=30):
    """Listens for state changed events on the listener and makes sure they match what we
    expect. Stop-and-restart events (where GetRestartedFromEvent() returns true) are ignored."""

    for expected_state in states:
        def get_next_event():
            event = lldb.SBEvent()
            if not listener.WaitForEventForBroadcasterWithType(
                    timeout,
                    process.GetBroadcaster(),
                    lldb.SBProcess.eBroadcastBitStateChanged,
                    event):
                test.fail(
                    "Timed out while waiting for a transition to state %s" %
                    lldb.SBDebugger.StateAsCString(expected_state))
            return event

        event = get_next_event()
        while (lldb.SBProcess.GetStateFromEvent(event) == lldb.eStateStopped and
                lldb.SBProcess.GetRestartedFromEvent(event)):
            # Ignore restarted event and the subsequent running event.
            event = get_next_event()
            test.assertEqual(
                lldb.SBProcess.GetStateFromEvent(event),
                lldb.eStateRunning,
                "Restarted event followed by a running event")
            event = get_next_event()

        test.assertEqual(
            lldb.SBProcess.GetStateFromEvent(event),
            expected_state)

# ===================================
# Utility functions related to Frames
# ===================================


def get_parent_frame(frame):
    """
    Returns the parent frame of the input frame object; None if not available.
    """
    thread = frame.GetThread()
    parent_found = False
    for f in thread:
        if parent_found:
            return f
        if f.GetFrameID() == frame.GetFrameID():
            parent_found = True

    # If we reach here, no parent has been found, return None.
    return None


def get_args_as_string(frame, showFuncName=True):
    """
    Returns the args of the input frame object as a string.
    """
    # arguments     => True
    # locals        => False
    # statics       => False
    # in_scope_only => True
    vars = frame.GetVariables(True, False, False, True)  # type of SBValueList
    args = []  # list of strings
    for var in vars:
        args.append("(%s)%s=%s" % (var.GetTypeName(),
                                   var.GetName(),
                                   var.GetValue()))
    if frame.GetFunction():
        name = frame.GetFunction().GetName()
    elif frame.GetSymbol():
        name = frame.GetSymbol().GetName()
    else:
        name = ""
    if showFuncName:
        return "%s(%s)" % (name, ", ".join(args))
    else:
        return "(%s)" % (", ".join(args))


def print_registers(frame, string_buffer=False):
    """Prints all the register sets of the frame."""

    output = SixStringIO() if string_buffer else sys.stdout

    print("Register sets for " + str(frame), file=output)

    registerSet = frame.GetRegisters()  # Return type of SBValueList.
    print("Frame registers (size of register set = %d):" %
          registerSet.GetSize(), file=output)
    for value in registerSet:
        #print(value, file=output)
        print("%s (number of children = %d):" %
              (value.GetName(), value.GetNumChildren()), file=output)
        for child in value:
            print(
                "Name: %s, Value: %s" %
                (child.GetName(),
                 child.GetValue()),
                file=output)

    if string_buffer:
        return output.getvalue()


def get_registers(frame, kind):
    """Returns the registers given the frame and the kind of registers desired.

    Returns None if there's no such kind.
    """
    registerSet = frame.GetRegisters()  # Return type of SBValueList.
    for value in registerSet:
        if kind.lower() in value.GetName().lower():
            return value

    return None


def get_GPRs(frame):
    """Returns the general purpose registers of the frame as an SBValue.

    The returned SBValue object is iterable.  An example:
        ...
        from lldbutil import get_GPRs
        regs = get_GPRs(frame)
        for reg in regs:
            print("%s => %s" % (reg.GetName(), reg.GetValue()))
        ...
    """
    return get_registers(frame, "general purpose")


def get_FPRs(frame):
    """Returns the floating point registers of the frame as an SBValue.

    The returned SBValue object is iterable.  An example:
        ...
        from lldbutil import get_FPRs
        regs = get_FPRs(frame)
        for reg in regs:
            print("%s => %s" % (reg.GetName(), reg.GetValue()))
        ...
    """
    return get_registers(frame, "floating point")


def get_ESRs(frame):
    """Returns the exception state registers of the frame as an SBValue.

    The returned SBValue object is iterable.  An example:
        ...
        from lldbutil import get_ESRs
        regs = get_ESRs(frame)
        for reg in regs:
            print("%s => %s" % (reg.GetName(), reg.GetValue()))
        ...
    """
    return get_registers(frame, "exception state")

# ======================================
# Utility classes/functions for SBValues
# ======================================


class BasicFormatter(object):
    """The basic formatter inspects the value object and prints the value."""

    def format(self, value, buffer=None, indent=0):
        if not buffer:
            output = SixStringIO()
        else:
            output = buffer
        # If there is a summary, it suffices.
        val = value.GetSummary()
        # Otherwise, get the value.
        if val is None:
            val = value.GetValue()
        if val is None and value.GetNumChildren() > 0:
            val = "%s (location)" % value.GetLocation()
        print("{indentation}({type}) {name} = {value}".format(
            indentation=' ' * indent,
            type=value.GetTypeName(),
            name=value.GetName(),
            value=val), file=output)
        return output.getvalue()


class ChildVisitingFormatter(BasicFormatter):
    """The child visiting formatter prints the value and its immediate children.

    The constructor takes a keyword arg: indent_child, which defaults to 2.
    """

    def __init__(self, indent_child=2):
        """Default indentation of 2 SPC's for the children."""
        self.cindent = indent_child

    def format(self, value, buffer=None):
        if not buffer:
            output = SixStringIO()
        else:
            output = buffer

        BasicFormatter.format(self, value, buffer=output)
        for child in value:
            BasicFormatter.format(
                self, child, buffer=output, indent=self.cindent)

        return output.getvalue()


class RecursiveDecentFormatter(BasicFormatter):
    """The recursive decent formatter prints the value and the decendents.

    The constructor takes two keyword args: indent_level, which defaults to 0,
    and indent_child, which defaults to 2.  The current indentation level is
    determined by indent_level, while the immediate children has an additional
    indentation by inden_child.
    """

    def __init__(self, indent_level=0, indent_child=2):
        self.lindent = indent_level
        self.cindent = indent_child

    def format(self, value, buffer=None):
        if not buffer:
            output = SixStringIO()
        else:
            output = buffer

        BasicFormatter.format(self, value, buffer=output, indent=self.lindent)
        new_indent = self.lindent + self.cindent
        for child in value:
            if child.GetSummary() is not None:
                BasicFormatter.format(
                    self, child, buffer=output, indent=new_indent)
            else:
                if child.GetNumChildren() > 0:
                    rdf = RecursiveDecentFormatter(indent_level=new_indent)
                    rdf.format(child, buffer=output)
                else:
                    BasicFormatter.format(
                        self, child, buffer=output, indent=new_indent)

        return output.getvalue()

# ===========================================================
# Utility functions for path manipulation on remote platforms
# ===========================================================


def join_remote_paths(*paths):
    # TODO: update with actual platform name for remote windows once it exists
    if lldb.remote_platform.GetName() == 'remote-windows':
        return os.path.join(*paths).replace(os.path.sep, '\\')
    return os.path.join(*paths).replace(os.path.sep, '/')


def append_to_process_working_directory(test, *paths):
    remote = lldb.remote_platform
    if remote:
        return join_remote_paths(remote.GetWorkingDirectory(), *paths)
    return os.path.join(test.getBuildDir(), *paths)

# ==================================================
# Utility functions to get the correct signal number
# ==================================================

import signal


def get_signal_number(signal_name):
    platform = lldb.remote_platform
    if platform and platform.IsValid():
        signals = platform.GetUnixSignals()
        if signals.IsValid():
            signal_number = signals.GetSignalNumberFromName(signal_name)
            if signal_number > 0:
                return signal_number
    # No remote platform; fall back to using local python signals.
    return getattr(signal, signal_name)


class PrintableRegex(object):

    def __init__(self, text):
        self.regex = re.compile(text)
        self.text = text

    def match(self, str):
        return self.regex.match(str)

    def __str__(self):
        return "%s" % (self.text)

    def __repr__(self):
        return "re.compile(%s) -> %s" % (self.text, self.regex)


def skip_if_callable(test, mycallable, reason):
    if six.callable(mycallable):
        if mycallable(test):
            test.skipTest(reason)
            return True
    return False


def skip_if_library_missing(test, target, library):
    def find_library(target, library):
        for module in target.modules:
            filename = module.file.GetFilename()
            if isinstance(library, str):
                if library == filename:
                    return False
            elif hasattr(library, 'match'):
                if library.match(filename):
                    return False
        return True

    def find_library_callable(test):
        return find_library(target, library)
    return skip_if_callable(
        test,
        find_library_callable,
        "could not find library matching '%s' in target %s" %
        (library,
         target))


def read_file_on_target(test, remote):
    if lldb.remote_platform:
        local = test.getBuildArtifact("file_from_target")
        error = lldb.remote_platform.Get(lldb.SBFileSpec(remote, False),
                    lldb.SBFileSpec(local, True))
        test.assertTrue(error.Success(), "Reading file {0} failed: {1}".format(remote, error))
    else:
        local = remote
    with open(local, 'r') as f:
        return f.read()

def read_file_from_process_wd(test, name):
    path = append_to_process_working_directory(test, name)
    return read_file_on_target(test, path)

def wait_for_file_on_target(testcase, file_path, max_attempts=6):
    for i in range(max_attempts):
        err, retcode, msg = testcase.run_platform_command("ls %s" % file_path)
        if err.Success() and retcode == 0:
            break
        if i < max_attempts:
            # Exponential backoff!
            import time
            time.sleep(pow(2, i) * 0.25)
    else:
        testcase.fail(
            "File %s not found even after %d attempts." %
            (file_path, max_attempts))

    return read_file_on_target(testcase, file_path)

def packetlog_get_process_info(log):
    """parse a gdb-remote packet log file and extract the response to qProcessInfo"""
    process_info = dict()
    with open(log, "r") as logfile:
        process_info_ostype = None
        expect_process_info_response = False
        for line in logfile:
            if expect_process_info_response:
                for pair in line.split(';'):
                    keyval = pair.split(':')
                    if len(keyval) == 2:
                        process_info[keyval[0]] = keyval[1]
                break
            if 'send packet: $qProcessInfo#' in line:
                expect_process_info_response = True
    return process_info

def packetlog_get_dylib_info(log):
    """parse a gdb-remote packet log file and extract the *last* complete
    (=> fetch_all_solibs=true) response to jGetLoadedDynamicLibrariesInfos"""
    import json
    dylib_info = None
    with open(log, "r") as logfile:
        dylib_info = None
        expect_dylib_info_response = False
        for line in logfile:
            if expect_dylib_info_response:
                while line[0] != '$':
                    line = line[1:]
                line = line[1:]
                # Unescape '}'.
                dylib_info = json.loads(line.replace('}]','}')[:-4])
                expect_dylib_info_response = False
            if 'send packet: $jGetLoadedDynamicLibrariesInfos:{"fetch_all_solibs":true}' in line:
                expect_dylib_info_response = True

    return dylib_info
