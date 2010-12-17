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

class SBThread
{
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

    // Get the number of words associated with the stop reason.
    size_t
    GetStopReasonDataCount();

    //--------------------------------------------------------------------------
    // Get information associated with a stop reason.
    //
    // Breakpoint stop reasons will have data that consists of pairs of 
    // breakpoint IDs followed by the breakpoint location IDs (they always come
    // in pairs).
    //
    // Stop Reason              Count Data Type
    // ======================== ===== ==========================================
    // eStopReasonNone          0
    // eStopReasonTrace         0
    // eStopReasonBreakpoint    N     duple: {breakpoint id, location id}
    // eStopReasonWatchpoint    N     duple: {watchpoint id, location id}
    // eStopReasonSignal        1     unix signal number
    // eStopReasonException     N     exception data
    // eStopReasonPlanComplete  0
    //--------------------------------------------------------------------------
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
    StepInstruction(bool step_over);

    void
    RunToAddress (lldb::addr_t addr);

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
