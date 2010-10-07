"""
This LLDB module contains miscellaneous utilities.
"""

import lldb
import sys
import StringIO

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

    print >> output, "Stack trace for thread id={0:#x} name={1} queue={2}:".format(
        thread.GetThreadID(), thread.GetName(), thread.GetQueueName())

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
