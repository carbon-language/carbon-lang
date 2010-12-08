"""
This LLDB module contains miscellaneous utilities.
"""

import lldb
import sys
import StringIO

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


# =================================================
# Convert some enum value to its string counterpart
# =================================================

def StateTypeString(enum):
    """Returns the stateType string given an enum."""
    if enum == lldb.eStateInvalid:
        return "invalid"
    elif enum == lldb.eStateUnloaded:
        return "unloaded"
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
        raise Exception("Unknown stopReason enum")

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
        raise Exception("Unknown stopReason enum")

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
        raise Exception("Unknown valueType enum")


# ==================================================
# Utility functions related to Threads and Processes
# ==================================================

def GetFunctionNames(thread):
    """
    Returns a sequence of function names from the stack frames of this thread.
    """
    def GetFuncName(i):
        return thread.GetFrameAtIndex(i).GetFunction().GetName()

    return map(GetFuncName, range(thread.GetNumFrames()))


def GetSymbolNames(thread):
    """
    Returns a sequence of symbols for this thread.
    """
    def GetSymbol(i):
        return thread.GetFrameAtIndex(i).GetSymbol().GetName()

    return map(GetSymbol, range(thread.GetNumFrames()))


def GetPCAddresses(thread):
    """
    Returns a sequence of pc addresses for this thread.
    """
    def GetPCAddress(i):
        return thread.GetFrameAtIndex(i).GetPCAddress()

    return map(GetPCAddress, range(thread.GetNumFrames()))


def GetFilenames(thread):
    """
    Returns a sequence of file names from the stack frames of this thread.
    """
    def GetFilename(i):
        return thread.GetFrameAtIndex(i).GetLineEntry().GetFileSpec().GetFilename()

    return map(GetFilename, range(thread.GetNumFrames()))


def GetLineNumbers(thread):
    """
    Returns a sequence of line numbers from the stack frames of this thread.
    """
    def GetLineNumber(i):
        return thread.GetFrameAtIndex(i).GetLineEntry().GetLine()

    return map(GetLineNumber, range(thread.GetNumFrames()))


def GetModuleNames(thread):
    """
    Returns a sequence of module names from the stack frames of this thread.
    """
    def GetModuleName(i):
        return thread.GetFrameAtIndex(i).GetModule().GetFileSpec().GetFilename()

    return map(GetModuleName, range(thread.GetNumFrames()))


def GetStackFrames(thread):
    """
    Returns a sequence of stack frames for this thread.
    """
    def GetStackFrame(i):
        return thread.GetFrameAtIndex(i)

    return map(GetStackFrame, range(thread.GetNumFrames()))


def PrintStackTrace(thread, string_buffer = False):
    """Prints a simple stack trace of this thread."""

    output = StringIO.StringIO() if string_buffer else sys.stdout
    target = thread.GetProcess().GetTarget()

    depth = thread.GetNumFrames()

    mods = GetModuleNames(thread)
    funcs = GetFunctionNames(thread)
    symbols = GetSymbolNames(thread)
    files = GetFilenames(thread)
    lines = GetLineNumbers(thread)
    addrs = GetPCAddresses(thread)

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


def PrintStackTraces(process, string_buffer = False):
    """Prints the stack traces of all the threads."""

    output = StringIO.StringIO() if string_buffer else sys.stdout

    print >> output, "Stack traces for " + repr(process)

    for i in range(process.GetNumThreads()):
        print >> output, PrintStackTrace(process.GetThreadAtIndex(i), string_buffer=True)

    if string_buffer:
        return output.getvalue()
