#!/usr/bin/python

#----------------------------------------------------------------------
# Be sure to add the python path that points to the LLDB shared library.
# On MacOSX csh, tcsh:
#   setenv PYTHONPATH /Applications/Xcode.app/Contents/SharedFrameworks/LLDB.framework/Resources/Python
# On MacOSX sh, bash:
#   export PYTHONPATH=/Applications/Xcode.app/Contents/SharedFrameworks/LLDB.framework/Resources/Python
#----------------------------------------------------------------------

import commands
import optparse
import os
import platform
import re
import resource
import sys
import time

#----------------------------------------------------------------------
# Code that auto imports LLDB
#----------------------------------------------------------------------
try: 
    # Just try for LLDB in case PYTHONPATH is already correctly setup
    import lldb
except ImportError:
    lldb_python_dirs = list()
    # lldb is not in the PYTHONPATH, try some defaults for the current platform
    platform_system = platform.system()
    if platform_system == 'Darwin':
        # On Darwin, try the currently selected Xcode directory
        xcode_dir = commands.getoutput("xcode-select --print-path")
        if xcode_dir:
            lldb_python_dirs.append(os.path.realpath(xcode_dir + '/../SharedFrameworks/LLDB.framework/Resources/Python'))
            lldb_python_dirs.append(xcode_dir + '/Library/PrivateFrameworks/LLDB.framework/Resources/Python')
        lldb_python_dirs.append('/System/Library/PrivateFrameworks/LLDB.framework/Resources/Python')
    success = False
    for lldb_python_dir in lldb_python_dirs:
        if os.path.exists(lldb_python_dir):
            if not (sys.path.__contains__(lldb_python_dir)):
                sys.path.append(lldb_python_dir)
                try: 
                    import lldb
                except ImportError:
                    pass
                else:
                    print 'imported lldb from: "%s"' % (lldb_python_dir)
                    success = True
                    break
    if not success:
        print "error: couldn't locate the 'lldb' module, please set PYTHONPATH correctly"
        sys.exit(1)


class Timer:    
    def __enter__(self):
        self.start = time.clock()
        return self

    def __exit__(self, *args):
        self.end = time.clock()
        self.interval = self.end - self.start

class TestCase:
    """Class that aids in running performance tests."""
    def __init__(self):
        self.verbose = False
        self.debugger = lldb.SBDebugger.Create()
        self.target = None
        self.process = None
        self.thread = None
        self.launch_info = None
        self.listener = self.debugger.GetListener()
    
    def Setup(self, args):
        self.launch_info = lldb.SBLaunchInfo(args)
    
    def Run (self, args):
        assert False, "performance.TestCase.Run() must be subclassed"
        
    def Launch(self):
        if self.target:
            error = lldb.SBError()
            self.process = self.target.Launch (self.launch_info, error);
            if not error.Success():
                print "error: %s" % error.GetCString()
            if self.process:
                self.process.GetBroadcaster().AddListener(self.listener, lldb.SBProcess.eBroadcastBitStateChanged | lldb.SBProcess.eBroadcastBitInterrupt);
                return True
        return False
        
    def WaitForNextProcessEvent (self):
        event = None
        if self.process:
            while event is None:
                process_event = lldb.SBEvent()
                if self.listener.WaitForEvent (lldb.UINT32_MAX, process_event):
                    state = lldb.SBProcess.GetStateFromEvent (process_event)
                    if self.verbose:
                        print "event = %s" % (lldb.SBDebugger.StateAsCString(state))
                    if lldb.SBProcess.GetRestartedFromEvent(process_event):
                        continue
                    if state == lldb.eStateInvalid or state == lldb.eStateDetached or state == lldb.eStateCrashed or  state == lldb.eStateUnloaded or state == lldb.eStateExited:
                       event = process_event
                    elif state == lldb.eStateConnected or state == lldb.eStateAttaching or state == lldb.eStateLaunching or state == lldb.eStateRunning or state == lldb.eStateStepping or state == lldb.eStateSuspended:
                         continue
                    elif state == lldb.eStateStopped:
                        event = process_event
                        call_test_step = True
                        fatal = False
                        selected_thread = False
                        for thread in self.process:
                            frame = thread.GetFrameAtIndex(0)
                            select_thread = False
                            stop_reason = thread.GetStopReason();
                            if self.verbose:
                                print "tid = %#x pc = %#x " % (thread.GetThreadID(),frame.GetPC()),
                            if stop_reason == lldb.eStopReasonNone:
                                if self.verbose:
                                    print "none"
                            elif stop_reason == lldb.eStopReasonTrace:
                                select_thread = True
                                if self.verbose:
                                    print "trace"
                            elif stop_reason == lldb.eStopReasonPlanComplete:
                                select_thread = True
                                if self.verbose:
                                    print "plan complete"
                            elif stop_reason == lldb.eStopReasonThreadExiting:
                                if self.verbose:
                                    print "thread exiting"
                            elif stop_reason == lldb.eStopReasonExec:
                                if self.verbose:
                                    print "exec"
                            elif stop_reason == lldb.eStopReasonInvalid:
                                if self.verbose:
                                    print "invalid"
                            elif stop_reason == lldb.eStopReasonException:
                                select_thread = True
                                if self.verbose:
                                    print "exception"
                                fatal = True
                            elif stop_reason == lldb.eStopReasonBreakpoint:
                                select_thread = True
                                if self.verbose:
                                    print "breakpoint id = %d.%d" % (thread.GetStopReasonDataAtIndex(0),thread.GetStopReasonDataAtIndex(1))
                            elif stop_reason == lldb.eStopReasonWatchpoint:
                                select_thread = True
                                if self.verbose:
                                    print "watchpoint id = %d" % (thread.GetStopReasonDataAtIndex(0))
                            elif stop_reason == lldb.eStopReasonSignal:
                                select_thread = True
                                if self.verbose:
                                    print "signal %d" % (thread.GetStopReasonDataAtIndex(0))

                            if select_thread and not selected_thread:
                                self.thread = thread;
                                selected_thread = self.process.SetSelectedThread(thread);
                        if fatal:
                            # if self.verbose: 
                            #     Xcode.RunCommand(self.debugger,"bt all",true);
                            sys.exit(1);
        return event
    
class Measurement:
    '''A class that encapsulates a measurement'''
    def Measure(self):
        assert False, "performance.Measurement.Measure() must be subclassed"
        
class MemoryMeasurement(Measurement):
    '''A class that can measure memory statistics for a process.'''
    def __init__(self, pid):
        self.pid = pid
        self.stats = ["rprvt","rshrd","rsize","vsize","vprvt","kprvt","kshrd","faults","cow","pageins"]
        self.command = "top -l 1 -pid %u -stats %s" % (self.pid, ",".join(self.stats))
        self.value = dict()
    
    def Measure(self):
        output = commands.getoutput(self.command).split("\n")[-1]
        values = re.split('[-+\s]+', output)
        for (idx, stat) in enumerate(values):
            multiplier = 1
            if stat:
                if stat[-1] == 'K':
                    multiplier = 1024;
                    stat = stat[:-1]
                elif stat[-1] == 'M':
                    multiplier = 1024*1024;
                    stat = stat[:-1]
                elif stat[-1] == 'G':
                    multiplier = 1024*1024*1024;
                elif stat[-1] == 'T':
                    multiplier = 1024*1024*1024*1024;
                    stat = stat[:-1]
                self.value[self.stats[idx]] = int (stat) * multiplier

    def __str__(self):
        '''Dump the MemoryMeasurement current value'''
        s = ''
        for key in self.value.keys():
            if s:
                s += "\n"
            s += "%8s = %s" % (key, self.value[key])
        return s


class TesterTestCase(TestCase):
    
    def Run (self, args):
        self.Setup(args)
        self.verbose = True
        #self.breakpoints = { 'name' : { 'main' } : , 'malloc' {}
        with Timer() as total_time:
            self.target = self.debugger.CreateTarget(args[0])
            if self.target:
                with Timer() as breakpoint_timer:
                    self.target.BreakpointCreateByName("main")
                print('Breakpoint time = %.03f sec.' % breakpoint_timer.interval)
                if self.Launch():
                    self.WaitForNextProcessEvent();
                    self.process.Kill()
                else:
                    print "error: failed to launch process"
            else:
                print "error: failed to create target with '%s'" % (args[0])
        print('Total time = %.03f sec.' % total_time.interval)
        

if __name__ == '__main__':
    lldb.SBDebugger.Initialize()
    test = TesterTestCase()
    test.Run (sys.argv[1:])
    mem = MemoryMeasurement(os.getpid())
    mem.Measure()
    print str(mem)
    lldb.SBDebugger.Terminate()
    # print "sleeeping for 100 seconds"
    # time.sleep(100)
