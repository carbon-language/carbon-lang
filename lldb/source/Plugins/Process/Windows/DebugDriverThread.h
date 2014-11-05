//===-- DebugDriverThread.h -------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_Plugins_Process_Windows_DebugDriverThread_H_
#define liblldb_Plugins_Process_Windows_DebugDriverThread_H_

#include "IDebugEventHandler.h"

#include "lldb/Host/HostThread.h"
#include "lldb/Host/windows/windows.h"
#include "lldb/lldb-types.h"

#include <map>

class ProcessWindows;

namespace lldb_private
{
class DriverMessage;
class DriverMessageResult;
class DriverLaunchProcessMessage;
class DriverLaunchProcessMessageResult;

class DebugOneProcessThread;

//----------------------------------------------------------------------
// DebugDriverThread
//
// Runs a background thread that pumps a queue from the application to tell the
// debugger to do different things like launching processes, attaching to
// processes, etc.
//----------------------------------------------------------------------
class DebugDriverThread : public IDebugEventHandler
{
    friend class DebugOneProcessThread;

  public:
    virtual ~DebugDriverThread();

    static void Initialize();
    static void Teardown();
    static DebugDriverThread &GetInstance();

    void PostDebugMessage(const DriverMessage *message);

  private:
    DebugDriverThread();

    void Shutdown();

    bool ProcessDriverMessages();

    const DriverMessageResult *HandleDriverMessage(const DriverMessage *message);
    const DriverLaunchProcessMessageResult *HandleDriverMessage(const DriverLaunchProcessMessage *launch_message);

    // Debug event handlers.  These are invoked on the driver thread by way of QueueUserAPC as
    // events happen in the inferiors.
    virtual void OnProcessLaunched(const ProcessMessageCreateProcess &message) override;
    virtual void OnExitProcess(const ProcessMessageExitProcess &message) override;
    virtual void OnDebuggerConnected(const ProcessMessageDebuggerConnected &message) override;
    virtual void OnDebugException(const ProcessMessageException &message) override;
    virtual void OnCreateThread(const ProcessMessageCreateThread &message) override;
    virtual void OnExitThread(const ProcessMessageExitThread &message) override;
    virtual void OnLoadDll(const ProcessMessageLoadDll &message) override;
    virtual void OnUnloadDll(const ProcessMessageUnloadDll &message) override;
    virtual void OnDebugString(const ProcessMessageDebugString &message) override;
    virtual void OnDebuggerError(const ProcessMessageDebuggerError &message) override;

    static DebugDriverThread *m_instance;

    std::map<lldb::pid_t, std::shared_ptr<DebugOneProcessThread>> m_debugged_processes;

    HANDLE m_driver_message_event;
    HANDLE m_shutdown_event;
    HANDLE m_driver_pipe_read;
    HANDLE m_driver_pipe_write;
    lldb_private::HostThread m_driver_thread;

    static lldb::thread_result_t DriverThread(void *data);
};
}

#endif
