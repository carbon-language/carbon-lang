//===-- LinuxThread.h -------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_LinuxThread_H_
#define liblldb_LinuxThread_H_

// C Includes
// C++ Includes
#include <memory>

// Other libraries and framework includes
#include "lldb/Target/Thread.h"

class ProcessMonitor;
class RegisterContextLinux;

//------------------------------------------------------------------------------
// @class LinuxThread
// @brief Abstraction of a linux process (thread).
class LinuxThread
    : public lldb_private::Thread
{
public:
    LinuxThread(lldb_private::Process &process, lldb::tid_t tid);

    virtual ~LinuxThread();

    void
    RefreshStateAfterStop();

    bool
    WillResume(lldb::StateType resume_state);

    const char *
    GetInfo();

    virtual lldb::RegisterContextSP
    GetRegisterContext();

    virtual lldb::RegisterContextSP
    CreateRegisterContextForFrame (lldb_private::StackFrame *frame);

    //--------------------------------------------------------------------------
    // These methods form a specialized interface to linux threads.
    //
    bool Resume();

    void BreakNotify();
    void TraceNotify();
    void ExitNotify();

protected:
    virtual bool
    SaveFrameZeroState(RegisterCheckpoint &checkpoint);

    virtual bool
    RestoreSaveFrameZero(const RegisterCheckpoint &checkpoint);

private:
    
    RegisterContextLinux *
    GetRegisterContextLinux ()
    {
        if (!m_reg_context_sp)
            GetRegisterContext();
        return (RegisterContextLinux *)m_reg_context_sp.get();
    }
    
    std::auto_ptr<lldb_private::StackFrame> m_frame_ap;

    lldb::BreakpointSiteSP m_breakpoint;
    lldb::StopInfoSP m_stop_info;

    // Cached process stop id.  Used to ensure we do not recalculate stop
    // information/state needlessly.
    uint32_t m_stop_info_id;

    enum Notification {
        eNone,
        eBreak,
        eTrace,
        eExit
    };

    Notification m_note;

    ProcessMonitor &GetMonitor();

    lldb::StopInfoSP
    GetPrivateStopReason();

    void
    RefreshPrivateStopReason();

    lldb_private::Unwind *
    GetUnwinder();
};

#endif // #ifndef liblldb_LinuxThread_H_
