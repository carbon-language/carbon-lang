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

    virtual lldb::RegisterContextSP
    GetRegisterContext ();

    virtual lldb::RegisterContextSP
    CreateRegisterContextForFrame (lldb_private::StackFrame *frame);

    virtual bool
    SaveFrameZeroState (RegisterCheckpoint &checkpoint);

    virtual bool
    RestoreSaveFrameZero (const RegisterCheckpoint &checkpoint);

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

    void
    SetStopInfo (const lldb::StopInfoSP &stop_info)
    {
        m_actual_stop_info_sp = stop_info;
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
    
    friend class ProcessGDBRemote;

    void
    PrivateSetRegisterValue (uint32_t reg, 
                             StringExtractor &response);
                             
    //------------------------------------------------------------------
    // Member variables.
    //------------------------------------------------------------------
    std::string m_thread_name;
    std::string m_dispatch_queue_name;
    lldb::addr_t m_thread_dispatch_qaddr;
    uint32_t m_thread_stop_reason_stop_id;
    //------------------------------------------------------------------
    // Member variables.
    //------------------------------------------------------------------

    virtual lldb_private::Unwind *
    GetUnwinder ();

    void
    SetStopInfoFromPacket (StringExtractor &stop_packet, uint32_t stop_id);

    virtual lldb::StopInfoSP
    GetPrivateStopReason ();


};

#endif  // liblldb_ThreadGDBRemote_h_
