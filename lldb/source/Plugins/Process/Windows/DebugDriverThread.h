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

#include "lldb/Host/HostThread.h"
#include "lldb/Host/windows/windows.h"
#include "lldb/lldb-types.h"

#include <map>

class ProcessWindows;

namespace lldb_private
{
class DebugMonitorMessage;
class DebugMonitorMessageResult;
class DebugOneProcessThread;
class LaunchProcessMessage;
class LaunchProcessMessageResult;

class SlaveMessageProcessExited;
class SlaveMessageRipEvent;

//----------------------------------------------------------------------
// DebugDriverThread
//
// Runs a background thread that pumps a queue from the application to tell the
// debugger to do different things like launching processes, attaching to
// processes, etc.
//----------------------------------------------------------------------
class DebugDriverThread
{
    friend class DebugOneProcessThread;

  public:
    virtual ~DebugDriverThread();

    static void Initialize();
    static void Teardown();
    static DebugDriverThread &GetInstance();

    void PostDebugMessage(const DebugMonitorMessage *message);

  private:
    DebugDriverThread();

    void Shutdown();

    bool ProcessMonitorMessages();
    const DebugMonitorMessageResult *HandleMonitorMessage(const DebugMonitorMessage *message);
    const LaunchProcessMessageResult *HandleMonitorMessage(const LaunchProcessMessage *launch_message);

    // Slave message handlers.  These are invoked by the
    void HandleSlaveEvent(const SlaveMessageProcessExited &message);
    void HandleSlaveEvent(const SlaveMessageRipEvent &message);

    static DebugDriverThread *m_instance;

    std::map<lldb::pid_t, std::shared_ptr<DebugOneProcessThread>> m_debugged_processes;

    HANDLE m_monitor_event;
    HANDLE m_shutdown_event;
    HANDLE m_monitor_pipe_read;
    HANDLE m_monitor_pipe_write;
    lldb_private::HostThread m_monitor_thread;

    static lldb::thread_result_t MonitorThread(void *data);
};
}

#endif
