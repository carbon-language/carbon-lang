//===-- ThreadGDBRemote.h ---------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_ThreadGDBRemote_h_
#define liblldb_ThreadGDBRemote_h_

#include <string>

#include "lldb/Target/Process.h"
#include "lldb/Target/Thread.h"
#include "MachException.h"
#include "libunwind.h"

class StringExtractor;
class ProcessGDBRemote;

class ThreadGDBRemote : public lldb_private::Thread
{
public:
    ThreadGDBRemote (ProcessGDBRemote &process, lldb::tid_t tid);

    virtual
    ~ThreadGDBRemote ();

    virtual bool
    WillResume (lldb::StateType resume_state);

    virtual void
    RefreshStateAfterStop();

    virtual const char *
    GetInfo ();

    virtual const char *
    GetName ();

    virtual const char *
    GetQueueName ();

    virtual lldb_private::RegisterContext *
    GetRegisterContext ();

    virtual lldb_private::RegisterContext *
    CreateRegisterContextForFrame (lldb_private::StackFrame *frame);

    virtual bool
    SaveFrameZeroState (RegisterCheckpoint &checkpoint);

    virtual bool
    RestoreSaveFrameZero (const RegisterCheckpoint &checkpoint);

    virtual uint32_t
    GetStackFrameCount();

    virtual lldb::StackFrameSP
    GetStackFrameAtIndex (uint32_t idx);

    virtual void
    ClearStackFrames ();

    ProcessGDBRemote &
    GetGDBProcess ()
    {
        return (ProcessGDBRemote &)m_process;
    }

    const ProcessGDBRemote &
    GetGDBProcess () const
    {
        return (ProcessGDBRemote &)m_process;
    }

    void
    Dump (lldb_private::Log *log, uint32_t index);

    static bool
    ThreadIDIsValid (lldb::tid_t thread);

    bool
    ShouldStop (bool &step_more);

    const char *
    GetBasicInfoAsString ();

    lldb_private::Thread::StopInfo &
    GetStopInfoRef ()
    {
        return m_stop_info;
    }

    uint32_t
    GetStopInfoStopID()
    {
        return m_stop_info_stop_id;
    }

    void
    SetStopInfoStopID (uint32_t stop_id)
    {
        m_stop_info_stop_id = stop_id;
    }

    void
    SetName (const char *name)
    {
        if (name && name[0])
            m_thread_name.assign (name);
        else
            m_thread_name.clear();
    }

    lldb::addr_t
    GetThreadDispatchQAddr ()
    {
        return m_thread_dispatch_qaddr;
    }

    void
    SetThreadDispatchQAddr (lldb::addr_t thread_dispatch_qaddr)
    {
        m_thread_dispatch_qaddr = thread_dispatch_qaddr;
    }

protected:
    //------------------------------------------------------------------
    // Member variables.
    //------------------------------------------------------------------
    uint32_t m_stop_info_stop_id;
    lldb_private::Thread::StopInfo m_stop_info;
    std::string m_thread_name;
    std::string m_dispatch_queue_name;
    lldb::addr_t m_thread_dispatch_qaddr;
    std::auto_ptr<lldb_private::Unwind> m_unwinder_ap;
    //------------------------------------------------------------------
    // Member variables.
    //------------------------------------------------------------------

    lldb_private::Unwind *
    GetUnwinder ();

    void
    SetStopInfoFromPacket (StringExtractor &stop_packet, uint32_t stop_id);

    virtual bool
    GetRawStopReason (StopInfo *stop_info);


};

#endif  // liblldb_ThreadGDBRemote_h_
