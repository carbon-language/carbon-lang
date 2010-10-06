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

def disassemble_instructions (insts):
    for i in range(0, insts.GetSize()):
        print insts.GetInstructionAtIndex(i)

# Initialize LLDB so we can use it
lldb.SBDebugger.Initialize()

# Create a new debugger instance
debugger = lldb.SBDebugger.Create()

# When we step or continue, don't return from the function until the process 
# stops. We do this by setting the async mode to false.
debugger.SetAsync (False)

# Create a target from a file and arch
target = debugger.CreateTargetWithFileAndArch (sys.argv[1], "x86_64")

if target.IsValid():
    # If the target is valid set a breakpoint at main
    main_bp = target.BreakpointCreateByName ("main", sys.argv[1]);
    
    # Launch the process. Since we specified synchronous mode, we won't return
    # from this function until we hit the breakpoint at main
    process = target.LaunchProcess (sys.argv[2:], [''], "/dev/stdout", 0, False)
    
    # Make sure the launch went ok
    if process.IsValid():
        # Print some simple process info
        print "process:", process, "\n"
        # Get the first thread
        thread = process.GetThreadAtIndex (0)
        if thread.IsValid():
            # Print some simple thread info
            print "thread: ", thread
            # Get the first frame
            frame = thread.GetFrameAtIndex (0)
            if frame.IsValid():
                # Print some simple frame info
                print "frame: ", frame
                function = frame.GetFunction()
                # See if we have debug info (a function)
                if function.IsValid():
                    # We do have a function, print some info for the function
                    print "function: ", function, "\n"
                    # Now get all instructions for this function and print them
                    insts = function.GetInstructions(target)
                    disassemble_instructions (insts)
                else:
                    # See if we have a symbol in the symbol table for where we stopped
                    symbol = frame.GetSymbol();
                    if symbol.IsValid():
                        # We do have a symbol, print some info for the symbol
                        print "symbol: ", symbol, "\n"
                        # Now get all instructions for this symbol and print them
                        insts = symbol.GetInstructions(target)
                        disassemble_instructions (insts)
        # Now continue to the program exit
        process.Continue()
        # When we return from the above function we will hopefully be at the
        # program exit. Print out some process info
        print "process:", process, "\n"
        

lldb.SBDebugger.Terminate()
