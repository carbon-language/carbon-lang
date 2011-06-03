//===-- ThreadMacOSX.h ------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_ThreadMacOSX_h_
#define liblldb_ThreadMacOSX_h_

#include <libproc.h>

#include "lldb/Target/Process.h"
#include "lldb/Target/Thread.h"
#include "MachException.h"

class ProcessMacOSX;
class MachThreadContext;

class ThreadMacOSX : public lldb_private::Thread
{
public:
    ThreadMacOSX (ProcessMacOSX &process, lldb::tid_t tid);

    virtual
    ~ThreadMacOSX ();

    virtual bool
    WillResume (lldb::StateType resume_state);

    virtual void
    RefreshStateAfterStop();

    virtual const char *
    GetInfo ();

    virtual const char *
    GetName ();

    virtual lldb::RegisterContextSP
    GetRegisterContext ();

    virtual lldb::RegisterContextSP
    CreateRegisterContextForFrame (lldb_private::StackFrame *frame);

    virtual void
    ClearStackFrames ();

    ProcessMacOSX &
    GetMacOSXProcess ()
    {
        return (ProcessMacOSX &)m_process;
    }

    const ProcessMacOSX &
    GetMacOSXProcess () const
    {
        return (ProcessMacOSX &)m_process;
    }

    void
    Dump (lldb_private::Log *log, uint32_t index);

    lldb::tid_t
    InferiorThreadID () const;

    static bool
    ThreadIDIsValid (lldb::tid_t thread);

    int32_t
    Resume ();

    int32_t
    Suspend ();

    int32_t
    GetSuspendCount () const { return m_suspend_count; }

    bool
    RestoreSuspendCount ();

    uint32_t
    SetHardwareBreakpoint (const lldb_private::BreakpointSite *bp);

    uint32_t
    SetHardwareWatchpoint (const lldb_private::WatchpointLocation *wp);

    bool
    ClearHardwareBreakpoint (const lldb_private::BreakpointSite *bp);

    bool
    ClearHardwareWatchpoint (const lldb_private::WatchpointLocation *wp);

    void
    ThreadWillResume (lldb::StateType resume_state);

    virtual void
    DidResume ();

    bool
    ShouldStop (bool &step_more);

    bool
    NotifyException (MachException::Data& exc);

    const MachException::Data&
    GetStopException () { return m_stop_exception; }

    const char *
    GetBasicInfoAsString ();

    size_t
    GetStackFrameData (std::vector<std::pair<lldb::addr_t, lldb::addr_t> >& fp_pc_pairs);

    virtual lldb::StopInfoSP
    GetPrivateStopReason ();

protected:
    bool
    GetIdentifierInfo ();

    const char *
    GetDispatchQueueName();

    static bool
    GetBasicInfo (lldb::tid_t threadID, struct thread_basic_info *basic_info);

    virtual lldb_private::Unwind *
    GetUnwinder ();

    //------------------------------------------------------------------
    // Member variables.
    //------------------------------------------------------------------
    std::vector<std::pair<lldb::addr_t, lldb::addr_t> > m_fp_pc_pairs;
    struct thread_basic_info        m_basic_info;       // Basic information for a thread used to see if a thread is valid
    std::string                     m_basic_info_string;// Basic thread info as a C string.
#ifdef THREAD_IDENTIFIER_INFO_COUNT
    thread_identifier_info_data_t   m_ident_info;
    struct proc_threadinfo          m_proc_threadinfo;
    std::string                     m_dispatch_queue_name;
#endif
    int32_t                         m_suspend_count;    // The current suspend count
    MachException::Data             m_stop_exception;   // The best exception that describes why this thread is stopped
    std::auto_ptr<MachThreadContext> m_context;         // The arch specific thread context for this thread (register state and more)

};

#endif  // liblldb_ThreadMacOSX_h_
