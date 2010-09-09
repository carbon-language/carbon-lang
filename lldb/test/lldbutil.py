"""
LLDB modules which contains miscellaneous utilities.
"""

import lldb

def GetFunctionNames(thread):
    """
    Returns a sequence of function names from the stack frames of this thread.
    """
    def GetFuncName(i):
        return thread.GetFrameAtIndex(i).GetFunction().GetName()

    return map(GetFuncName, range(thread.GetNumFrames()))


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


def PrintStackTrace(thread):
    """Prints a simple stack trace of this thread."""
    depth = thread.GetNumFrames()

    mods = GetModuleNames(thread)
    funcs = GetFunctionNames(thread)
    files = GetFilenames(thread)
    lines = GetLineNumbers(thread)

    print "Stack trace for thread id={0:#x} name={1} queue={2}:".format(
        thread.GetThreadID(), thread.GetName(), thread.GetQueueName())

    for i in range(depth - 1):
        print "  frame #{num}: {mod}`{func} at {file}:{line}".format(
            num=i, mod=mods[i], func=funcs[i], file=files[i], line=lines[i])

    print "  frame #{num}: {mod}`start".format(num=depth-1, mod=mods[depth-1])
