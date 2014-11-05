//===-- DebugOneProcessThread.h ---------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_Plugins_Process_Windows_DebugOneProcessThread_H_
#define liblldb_Plugins_Process_Windows_DebugOneProcessThread_H_

#include "lldb/Host/HostProcess.h"
#include "lldb/Host/HostThread.h"
#include "lldb/Host/Predicate.h"
#include "lldb/Host/windows/windows.h"

namespace lldb_private
{
class DriverLaunchProcessMessage;
class DriverLaunchProcessMessageResult;

//----------------------------------------------------------------------
// DebugOneProcessThread
//
// Debugs a single process, notifying the process plugin and/or the debugger
// driver thread as appropriate when interesting things occur.
//----------------------------------------------------------------------
class DebugOneProcessThread : public std::enable_shared_from_this<DebugOneProcessThread>
{
  public:
    DebugOneProcessThread(HostThread driver_thread);
    virtual ~DebugOneProcessThread();

    const DriverLaunchProcessMessageResult *DebugLaunch(const DriverLaunchProcessMessage *message);

  private:
    void DebugLoop();
    DWORD HandleExceptionEvent(const EXCEPTION_DEBUG_INFO &info, DWORD thread_id);
    DWORD HandleCreateThreadEvent(const CREATE_THREAD_DEBUG_INFO &info, DWORD thread_id);
    DWORD HandleCreateProcessEvent(const CREATE_PROCESS_DEBUG_INFO &info, DWORD thread_id);
    DWORD HandleExitThreadEvent(const EXIT_THREAD_DEBUG_INFO &info, DWORD thread_id);
    DWORD HandleExitProcessEvent(const EXIT_PROCESS_DEBUG_INFO &info, DWORD thread_id);
    DWORD HandleLoadDllEvent(const LOAD_DLL_DEBUG_INFO &info, DWORD thread_id);
    DWORD HandleUnloadDllEvent(const UNLOAD_DLL_DEBUG_INFO &info, DWORD thread_id);
    DWORD HandleODSEvent(const OUTPUT_DEBUG_STRING_INFO &info, DWORD thread_id);
    DWORD HandleRipEvent(const RIP_INFO &info, DWORD thread_id);

    static void __stdcall NotifySlaveProcessExited(ULONG_PTR message);
    static void __stdcall NotifySlaveRipEvent(ULONG_PTR message);

    // The main debug driver thread which is controlling this slave.
    lldb_private::HostThread m_driver_thread;
    Predicate<const DriverLaunchProcessMessageResult *> m_launch_predicate;
    lldb::ProcessSP m_process_plugin;
    HostProcess m_process;

    static lldb::thread_result_t DebugLaunchThread(void *data);
    lldb::thread_result_t DebugLaunchThread(const DriverLaunchProcessMessage *message);
};
}

#endif
