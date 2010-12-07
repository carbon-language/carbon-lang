#!/usr/bin/python

#----------------------------------------------------------------------
# Be sure to add the python path that points to the LLDB shared library.
# On MacOSX csh, tcsh:
#   setenv PYTHONPATH /Developer/Library/PrivateFrameworks/LLDB.framework/Resources/Python
# On MacOSX sh, bash:
#   export PYTHONPATH=/Developer/Library/PrivateFrameworks/LLDB.framework/Resources/Python
#----------------------------------------------------------------------

import lldb
import os
import sys
import time

def print_var_value (v, file, frame):
        if v.GetNumChildren() > 0:
            for c in range(v.GetNumChildren()):
                if v.GetChildAtIndex(c) is None:
                        file.write("None")
                else:
                        if (v.GetChildAtIndex(c).GetName()) is None:
                                file.write("None")
                        else:
                                file.write(v.GetChildAtIndex(c).GetName())
                                file.write('=')
                                print_var_value(v.GetChildAtIndex(c), file, frame)
                                file.write(',')
        else:
            if v.GetValue(frame) is None:
                file.write("None")
            else:
                file.write(v.GetValue(frame))


def print_vars (vars, fname, line, file, frame, target, thread):
    # disable this thread.
    count = thread.GetStopReasonDataCount()
    bid = 0
    tid = 0
    for i in range(count):
        id = thread.GetStopReasonDataAtIndex(i)
        bp = target.FindBreakpointByID(id)
        if bp.IsValid():
            if bp.IsEnabled() == True:
                    bid = bp.GetID()
                    tid = bp.GetThreadID()
                    bp.SetEnabled(False)
        else:
            bp_loc = bp.FindLocationByID(thread.GetStopReasonDataAtIndex(i+1))
            if bp_loc.IsValid():
                bid = bp_loc.GetBreakPoint().GetID()
                tid = bp_loc.ThreadGetID()
                # print " { ", bp_loc.ThreadGetID(), " : ", bp_loc.GetBreakPoint().GetID(), " }} "
                bp_loc.SetEnabled(False);

    for i in range(vars.GetSize()):
        file.write("#Argument ")
        file.write(fname)
        file.write(':')
        file.write(str(line))
        file.write(' ')
        file.write(str(tid))
        file.write(':')
        file.write(str(bid))
        file.write(' ')
        v = vars.GetValueAtIndex(i)
        file.write(v.GetName())
        file.write(' ')
        print_var_value (v, file, frame)
        file.write('\n')

def set_breakpoints (target, breakpoint_filename):
    f = open(breakpoint_filename, "r")
    lines = f.readlines()
    for l in range(len(lines)):
        c = lines[l].split()
        # print "setting break point - ", c
        bp = target.BreakpointCreateByLocation (str(c[0]), int(c[1]))
    f.close()

def stop_at_breakpoint (process):
    if process.IsValid():
        state = process.GetState()
        if state != lldb.eStateStopped:
            return lldb.eStateInvalid
        thread = process.GetThreadAtIndex(0)
        if thread.IsValid():
            if thread.GetStopReason() == lldb.eStopReasonBreakpoint:
                    return lldb.eStateStopped
            else:
                    return lldb.eStateInvalid
        else:
            return lldb.eStateInvalid
    else:
        return lldb.eStateInvalid

# Create a new debugger instance
debugger = lldb.SBDebugger.Create()

# When we step or continue, don't return from the function until the process 
# stops. We do this by setting the async mode to false.
debugger.SetAsync (False)

# Create a target from a file and arch
##print "Creating a target for '%s'" % sys.argv[1]

target = debugger.CreateTargetWithFileAndArch (sys.argv[1], lldb.LLDB_ARCH_DEFAULT)

if target.IsValid():
    #print "target is valid"
    set_breakpoints (target, sys.argv[2])
    #main_bp = target.BreakpointCreateByLocation ("byval-alignment.c", 11)
    #main_bp2 = target.BreakpointCreateByLocation ("byval-alignment.c", 20)

    ##print main_bp

    # Launch the process. Since we specified synchronous mode, we won't return
    # from this function until we hit the breakpoint at main
    process = target.LaunchProcess ([''], [''], "/dev/stdout", 0, False)
    file=open(str(sys.argv[3]), 'w')    
    # Make sure the launch went ok
    while stop_at_breakpoint(process) == lldb.eStateStopped:
        thread = process.GetThreadAtIndex (0)
        frame = thread.GetFrameAtIndex (0)
        if frame.IsValid():
            # #Print some simple frame info
            ##print frame
            #print "frame is valid"
            function = frame.GetFunction()
            if function.IsValid():
                fname = function.GetMangledName()
                if fname is None:
                    fname = function.GetName()
                #print "function : ",fname
                vars = frame.GetVariables(1,0,0,0)
                line = frame.GetLineEntry().GetLine()
                print_vars (vars, fname, line, file, frame, target, thread)
                #print vars
        process.Continue()
    file.close()

lldb.SBDebugger.Terminate()
