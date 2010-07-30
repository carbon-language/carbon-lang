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
    DisplayFramesForCurrentContext (FILE *out,
                                    FILE *err,
                                    uint32_t first_frame,
                                    uint32_t num_frames,
                                    bool show_frame_info,
                                    uint32_t num_frames_with_source,
                                    uint32_t source_lines_before = 3,
                                    uint32_t source_lines_after = 3);

    bool
    DisplaySingleFrameForCurrentContext (FILE *out,
                                         FILE *err,
                                         lldb::SBFrame &frame,
                                         bool show_frame_info,
                                         bool show_source,
                                         uint32_t source_lines_after,
                                         uint32_t source_lines_before);

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


protected:
    friend class SBBreakpoint;
    friend class SBBreakpointLocation;
    friend class SBFrame;
    friend class SBProcess;
    friend class SBDebugger;
    friend class SBValue;

    lldb_private::Thread *
    GetLLDBObjectPtr ();

#ifndef SWIG

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
