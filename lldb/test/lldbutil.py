"""
This LLDB module contains miscellaneous utilities.
"""

import lldb
import os, sys
import StringIO

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

# ===========================================
# Iterator for lldb aggregate data structures
# ===========================================

def lldb_iter(obj, getsize, getelem):
    """A generator adaptor for lldb aggregate data structures.

    API clients pass in an aggregate object or a container of it, the name of
    the method to get the size of the aggregate, and the name of the method to
    get the element by index.

    Example usages:

    1. Pass an aggregate as the first argument:

    def disassemble_instructions (insts):
        from lldbutil import lldb_iter
        for i in lldb_iter(insts, 'GetSize', 'GetInstructionAtIndex'):
            print i

    2. Pass a container of aggregate which provides APIs to get to the size and
       the element of the aggregate:

    # Module is a container of symbol table 
    module = target.FindModule(filespec)
    for symbol in lldb_iter(module, 'GetNumSymbols', 'GetSymbolAtIndex'):
        name = symbol.GetName()
        ...
    """
    size = getattr(obj, getsize)
    elem = getattr(obj, getelem)
    for i in range(size()):
        yield elem(i)


# ===================================================
# Disassembly for an SBFunction or an SBSymbol object
# ===================================================

def disassemble(target, function_or_symbol):
    """Disassemble the function or symbol given a target.

    It returns the disassembly content in a string object.
    """
    buf = StringIO.StringIO()
    insts = function_or_symbol.GetInstructions(target)
    for i in lldb_iter(insts, 'GetSize', 'GetInstructionAtIndex'):
        print >> buf, i
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
    return bytearray(map(ord, packed))

def bytearray_to_int(bytes, bytesize):
    """Utility function to convert a bytearray into an integer.

    It interprets the bytearray in the little endian format. For a big endian
    bytearray, just do ba.reverse() on the object before passing it in.
    """
    import struct

    if bytesize == 1:
        return ba[0]

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

    unpacked = struct.unpack(fmt, str(bytes))
    return unpacked[0]


# ==============================================================
# Get the description of an lldb object or None if not available
# ==============================================================
def get_description(obj, option=None):
    """Calls lldb_obj.GetDescription() and returns a string, or None.

    For SBTarget and SBBreakpointLocation lldb objects, an extra option can be
    passed in to describe the detailed level of description desired:
        o lldb.eDescriptionLevelBrief
        o lldb.eDescriptionLevelFull
        o lldb.eDescriptionLevelVerbose
    """
    method = getattr(obj, 'GetDescription')
    if not method:
        return None
    if isinstance(obj, lldb.SBTarget) or isinstance(obj, lldb.SBBreakpointLocation):
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

def StateTypeString(enum):
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

def StopReasonString(enum):
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
    elif enum == lldb.eStopReasonSignal:
        return "signal"
    elif enum == lldb.eStopReasonException:
        return "exception"
    elif enum == lldb.eStopReasonPlanComplete:
        return "plancomplete"
    else:
        raise Exception("Unknown StopReason enum")

def ValueTypeString(enum):
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
# Utility functions related to Threads and Processes
# ==================================================

def get_stopped_threads(process, reason):
    """Returns the thread(s) with the specified stop reason in a list."""
    threads = []
    for t in lldb_iter(process, 'GetNumThreads', 'GetThreadAtIndex'):
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
        thread = get_stopped_thread(self.process, lldb.eStopReasonPlanComplete)
        self.assertTrue(thread != None, "There should be a thread stopped due to breakpoint condition")
    ...

    2. Get the thread stopped due to a breakpoint

    ...
        from lldbutil import get_stopped_thread
        thread = get_stopped_thread(self.process, lldb.eStopReasonBreakpoint)
        self.assertTrue(thread != None, "There should be a thread stopped due to breakpoint")
    ...

    """
    threads = get_stopped_threads(process, reason)
    if len(threads) == 0:
        return None
    return threads[0]

def get_threads_stopped_at_breakpoint (process, bkpt):
    """ For a stopped process returns the thread stopped at the breakpoint passed in bkpt"""
    stopped_threads = []
    threads = []

    stopped_threads = get_stopped_threads (process, lldb.eStopReasonBreakpoint)

    if len(stopped_threads) == 0:
        return threads
    
    for thread in stopped_threads:
    # Make sure we've hit our breakpoint...
        break_id = thread.GetStopReasonDataAtIndex (0)
        if break_id == bkpt.GetID():
            threads.append(thread)

    return threads

def continue_to_breakpoint (process, bkpt):
    """ Continues the process, if it stops, returns the threads stopped at bkpt; otherwise, returns None"""
    process.Continue()
    if process.GetState() != lldb.eStateStopped:
        return None
    else:
        return get_threads_stopped_at_breakpoint (process, bkpt)

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
        return thread.GetFrameAtIndex(i).GetFunction().GetName()

    return map(GetFuncName, range(thread.GetNumFrames()))


def get_symbol_names(thread):
    """
    Returns a sequence of symbols for this thread.
    """
    def GetSymbol(i):
        return thread.GetFrameAtIndex(i).GetSymbol().GetName()

    return map(GetSymbol, range(thread.GetNumFrames()))


def get_pc_addresses(thread):
    """
    Returns a sequence of pc addresses for this thread.
    """
    def GetPCAddress(i):
        return thread.GetFrameAtIndex(i).GetPCAddress()

    return map(GetPCAddress, range(thread.GetNumFrames()))


def get_filenames(thread):
    """
    Returns a sequence of file names from the stack frames of this thread.
    """
    def GetFilename(i):
        return thread.GetFrameAtIndex(i).GetLineEntry().GetFileSpec().GetFilename()

    return map(GetFilename, range(thread.GetNumFrames()))


def get_line_numbers(thread):
    """
    Returns a sequence of line numbers from the stack frames of this thread.
    """
    def GetLineNumber(i):
        return thread.GetFrameAtIndex(i).GetLineEntry().GetLine()

    return map(GetLineNumber, range(thread.GetNumFrames()))


def get_module_names(thread):
    """
    Returns a sequence of module names from the stack frames of this thread.
    """
    def GetModuleName(i):
        return thread.GetFrameAtIndex(i).GetModule().GetFileSpec().GetFilename()

    return map(GetModuleName, range(thread.GetNumFrames()))


def get_stack_frames(thread):
    """
    Returns a sequence of stack frames for this thread.
    """
    def GetStackFrame(i):
        return thread.GetFrameAtIndex(i)

    return map(GetStackFrame, range(thread.GetNumFrames()))


def print_stacktrace(thread, string_buffer = False):
    """Prints a simple stack trace of this thread."""

    output = StringIO.StringIO() if string_buffer else sys.stdout
    target = thread.GetProcess().GetTarget()

    depth = thread.GetNumFrames()

    mods = get_module_names(thread)
    funcs = get_function_names(thread)
    symbols = get_symbol_names(thread)
    files = get_filenames(thread)
    lines = get_line_numbers(thread)
    addrs = get_pc_addresses(thread)

    if thread.GetStopReason() != lldb.eStopReasonInvalid:
        desc =  "stop reason=" + StopReasonString(thread.GetStopReason())
    else:
        desc = ""
    print >> output, "Stack trace for thread id={0:#x} name={1} queue={2} ".format(
        thread.GetThreadID(), thread.GetName(), thread.GetQueueName()) + desc

    for i in range(depth):
        frame = thread.GetFrameAtIndex(i)
        function = frame.GetFunction()

        load_addr = addrs[i].GetLoadAddress(target)
        if not function.IsValid():
            file_addr = addrs[i].GetFileAddress()
            print >> output, "  frame #{num}: {addr:#016x} {mod}`{symbol} + ????".format(
                num=i, addr=load_addr, mod=mods[i], symbol=symbols[i])
        else:
            print >> output, "  frame #{num}: {addr:#016x} {mod}`{func} at {file}:{line}".format(
                num=i, addr=load_addr, mod=mods[i], func=funcs[i], file=files[i], line=lines[i])

    if string_buffer:
        return output.getvalue()


def print_stacktraces(process, string_buffer = False):
    """Prints the stack traces of all the threads."""

    output = StringIO.StringIO() if string_buffer else sys.stdout

    print >> output, "Stack traces for " + repr(process)

    for i in range(process.GetNumThreads()):
        print >> output, print_stacktrace(process.GetThreadAtIndex(i), string_buffer=True)

    if string_buffer:
        return output.getvalue()
