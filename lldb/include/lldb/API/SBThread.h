//===-- SBThread.h ----------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SBThread_h_
#define LLDB_SBThread_h_

#include "lldb/API/SBDefines.h"

#include <stdio.h>

namespace lldb {

class SBFrame;

#ifdef SWIG
%feature("docstring",
"Represents a thread of execution. SBProcess contains SBThread(s).

SBThread supports frame iteration. For example (from test/python_api/
lldbutil/iter/TestLLDBIterator.py),

        from lldbutil import print_stacktrace
        stopped_due_to_breakpoint = False
        for thread in process:
            if self.TraceOn():
                print_stacktrace(thread)
            ID = thread.GetThreadID()
            if thread.GetStopReason() == lldb.eStopReasonBreakpoint:
                stopped_due_to_breakpoint = True
            for frame in thread:
                self.assertTrue(frame.GetThread().GetThreadID() == ID)
                if self.TraceOn():
                    print frame

        self.assertTrue(stopped_due_to_breakpoint)

See also SBProcess and SBFrame.
"
         ) SBThread;
#endif
class SBThread
{
#ifdef SWIG
    %feature("autodoc", "1");
#endif

public:
    SBThread ();

    SBThread (const lldb::SBThread &thread);

   ~SBThread();

    bool
    IsValid() const;

    void
    Clear ();

    lldb::StopReason
    GetStopReason();

#ifdef SWIG
    %feature("docstring", "
#endif
    /// Get the number of words associated with the stop reason.
    /// See also GetStopReasonDataAtIndex().
#ifdef SWIG
    ") GetStopReasonDataCount;
#endif
    size_t
    GetStopReasonDataCount();

#ifdef SWIG
    %feature("docstring", "
#endif
    //--------------------------------------------------------------------------
    /// Get information associated with a stop reason.
    ///
    /// Breakpoint stop reasons will have data that consists of pairs of 
    /// breakpoint IDs followed by the breakpoint location IDs (they always come
    /// in pairs).
    ///
    /// Stop Reason              Count Data Type
    /// ======================== ===== =========================================
    /// eStopReasonNone          0
    /// eStopReasonTrace         0
    /// eStopReasonBreakpoint    N     duple: {breakpoint id, location id}
    /// eStopReasonWatchpoint    N     duple: {watchpoint id, location id}
    /// eStopReasonSignal        1     unix signal number
    /// eStopReasonException     N     exception data
    /// eStopReasonPlanComplete  0
    //--------------------------------------------------------------------------
#ifdef SWIG
    ") GetStopReasonDataAtIndex;
#endif
    uint64_t
    GetStopReasonDataAtIndex(uint32_t idx);

    size_t
    GetStopDescription (char *dst, size_t dst_len);

    lldb::tid_t
    GetThreadID () const;

    uint32_t
    GetIndexID () const;

    const char *
    GetName () const;

    const char *
    GetQueueName() const;

    void
    StepOver (lldb::RunMode stop_other_threads = lldb::eOnlyDuringStepping);

    void
    StepInto (lldb::RunMode stop_other_threads = lldb::eOnlyDuringStepping);

    void
    StepOut ();

    void
    StepOutOfFrame (lldb::SBFrame &frame);

    void
    StepInstruction(bool step_over);

    SBError
    StepOverUntil (lldb::SBFrame &frame, 
                   lldb::SBFileSpec &file_spec, 
                   uint32_t line);

    void
    RunToAddress (lldb::addr_t addr);

#ifdef SWIG
    %feature("docstring", "
#endif
    //--------------------------------------------------------------------------
    /// LLDB currently supports process centric debugging which means when any
    /// thread in a process stops, all other threads are stopped. The Suspend()
    /// call here tells our process to suspend a thread and not let it run when
    /// the other threads in a process are allowed to run. So when 
    /// SBProcess::Continue() is called, any threads that aren't suspended will
    /// be allowed to run. If any of the SBThread functions for stepping are 
    /// called (StepOver, StepInto, StepOut, StepInstruction, RunToAddres), the
    /// thread will now be allowed to run and these funtions will simply return.
    ///
    /// Eventually we plan to add support for thread centric debugging where
    /// each thread is controlled individually and each thread would broadcast
    /// its state, but we haven't implemented this yet.
    /// 
    /// Likewise the SBThread::Resume() call will again allow the thread to run
    /// when the process is continued.
    ///
    /// Suspend() and Resume() functions are not currently reference counted, if
    /// anyone has the need for them to be reference counted, please let us
    /// know.
    //--------------------------------------------------------------------------
#ifdef SWIG
    ") Suspend;
#endif
    bool
    Suspend();
    
    bool
    Resume ();
    
    bool
    IsSuspended();

    uint32_t
    GetNumFrames ();

    lldb::SBFrame
    GetFrameAtIndex (uint32_t idx);

    lldb::SBFrame
    GetSelectedFrame ();

    lldb::SBFrame
    SetSelectedFrame (uint32_t frame_idx);

    lldb::SBProcess
    GetProcess ();

#ifndef SWIG

    const lldb::SBThread &
    operator = (const lldb::SBThread &rhs);

    bool
    operator == (const lldb::SBThread &rhs) const;

    bool
    operator != (const lldb::SBThread &rhs) const;

#endif

    bool
    GetDescription (lldb::SBStream &description) const;

protected:
    friend class SBBreakpoint;
    friend class SBBreakpointLocation;
    friend class SBFrame;
    friend class SBProcess;
    friend class SBDebugger;
    friend class SBValue;


#ifndef SWIG

    lldb_private::Thread *
    get ();

    const lldb_private::Thread *
    operator->() const;

    const lldb_private::Thread &
    operator*() const;


    lldb_private::Thread *
    operator->();

    lldb_private::Thread &
    operator*();

#endif

    SBThread (const lldb::ThreadSP& lldb_object_sp);

    void
    SetThread (const lldb::ThreadSP& lldb_object_sp);

private:
    //------------------------------------------------------------------
    // Classes that inherit from Thread can see and modify these
    //------------------------------------------------------------------

    lldb::ThreadSP m_opaque_sp;
};

} // namespace lldb

#endif  // LLDB_SBThread_h_
