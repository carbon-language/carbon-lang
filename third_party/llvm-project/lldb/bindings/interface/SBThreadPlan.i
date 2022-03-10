//===-- SBThread.h ----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SBThreadPlan_h_
#define LLDB_SBThreadPlan_h_

#include "lldb/API/SBDefines.h"

#include <stdio.h>

namespace lldb {

%feature("docstring",
"Represents a plan for the execution control of a given thread.

See also :py:class:`SBThread` and :py:class:`SBFrame`."
) SBThreadPlan;

class SBThreadPlan
{

friend class lldb_private::ThreadPlan;

public:
    SBThreadPlan ();

    SBThreadPlan (const lldb::SBThreadPlan &threadPlan);

    SBThreadPlan (const lldb::ThreadPlanSP& lldb_object_sp);

    SBThreadPlan (lldb::SBThread &thread, const char *class_name);

   ~SBThreadPlan ();

    bool
    IsValid();

    bool
    IsValid() const;

    explicit operator bool() const;

    void
    Clear ();

    lldb::StopReason
    GetStopReason();

    %feature("docstring", "
    Get the number of words associated with the stop reason.
    See also GetStopReasonDataAtIndex().") GetStopReasonDataCount;
    size_t
    GetStopReasonDataCount();

    %feature("docstring", "
    Get information associated with a stop reason.

    Breakpoint stop reasons will have data that consists of pairs of
    breakpoint IDs followed by the breakpoint location IDs (they always come
    in pairs).

    Stop Reason              Count Data Type
    ======================== ===== =========================================
    eStopReasonNone          0
    eStopReasonTrace         0
    eStopReasonBreakpoint    N     duple: {breakpoint id, location id}
    eStopReasonWatchpoint    1     watchpoint id
    eStopReasonSignal        1     unix signal number
    eStopReasonException     N     exception data
    eStopReasonExec          0
    eStopReasonFork          1     pid of the child process
    eStopReasonVFork         1     pid of the child process
    eStopReasonVForkDone     0
    eStopReasonPlanComplete  0") GetStopReasonDataAtIndex;
    uint64_t
    GetStopReasonDataAtIndex(uint32_t idx);

    SBThread
    GetThread () const;

    bool
    GetDescription (lldb::SBStream &description) const;

    void
    SetPlanComplete (bool success);

    bool
    IsPlanComplete();

    bool
    IsPlanStale();

    %feature("docstring", "Return whether this plan will ask to stop other threads when it runs.") GetStopOthers;
    bool
    GetStopOthers();

    %feature("docstring", "Set whether this plan will ask to stop other threads when it runs.")	GetStopOthers;
    void
    SetStopOthers(bool stop_others);

    SBThreadPlan
    QueueThreadPlanForStepOverRange (SBAddress &start_address,
                                     lldb::addr_t range_size);

    SBThreadPlan
    QueueThreadPlanForStepInRange (SBAddress &start_address,
                                   lldb::addr_t range_size);

    SBThreadPlan
    QueueThreadPlanForStepOut (uint32_t frame_idx_to_step_to, bool first_insn = false);

    SBThreadPlan
    QueueThreadPlanForRunToAddress (SBAddress address);

    SBThreadPlan
    QueueThreadPlanForStepScripted(const char *script_class_name);

    SBThreadPlan
    QueueThreadPlanForStepScripted(const char *script_class_name,
                                   SBError &error);
    SBThreadPlan
    QueueThreadPlanForStepScripted(const char *script_class_name,
                                   SBStructuredData &args_data,
                                   SBError &error);


protected:
    friend class SBBreakpoint;
    friend class SBBreakpointLocation;
    friend class SBFrame;
    friend class SBProcess;
    friend class SBDebugger;
    friend class SBValue;
    friend class lldb_private::QueueImpl;
    friend class SBQueueItem;

private:
    lldb::ThreadPlanSP m_opaque_sp;
};

} // namespace lldb

#endif  // LLDB_SBThreadPlan_h_
