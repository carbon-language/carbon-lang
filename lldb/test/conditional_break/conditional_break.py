import sys
import lldb
import lldbutil

def stop_if_called_from_a():
    dbg = lldb.SBDebugger.FindDebuggerWithID(lldb.debugger_unique_id)
    dbg.SetAsync(False)
    ci = dbg.GetCommandInterpreter()
    res = lldb.SBCommandReturnObject()

    target = dbg.GetSelectedTarget()
    process = target.GetProcess()
    thread = process.GetThreadAtIndex(0)

    print >> sys.stderr, "Checking call frames..."
    lldbutil.PrintStackTrace(thread)
    if thread.GetNumFrames() >= 2:
        funcs = lldbutil.GetFunctionNames(thread)
        print >> sys.stderr, funcs[0], "called from", funcs[1]
        if (funcs[0] == 'c' and funcs[1] == 'a'):
            print >> sys.stderr, "Stopped at c() with immediate caller as a()."
        else:
            print >> sys.stderr, "Continuing..."
            ci.HandleCommand("process continue", res)

    return True

