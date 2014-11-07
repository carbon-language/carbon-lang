//===-- DebuggerThread.h ----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_Plugins_Process_Windows_DebuggerThread_H_
#define liblldb_Plugins_Process_Windows_DebuggerThread_H_

#include "ForwardDecl.h"
#include "lldb/Host/HostProcess.h"
#include "lldb/Host/HostThread.h"
#include "lldb/Host/windows/windows.h"

#include <memory>

namespace lldb_private
{

//----------------------------------------------------------------------
// DebuggerThread
//
// Debugs a single process, notifying listeners as appropriate when interesting
// things occur.
//----------------------------------------------------------------------
class DebuggerThread : public std::enable_shared_from_this<DebuggerThread>
{
  public:
    DebuggerThread(DebugDelegateSP debug_delegate);
    virtual ~DebuggerThread();

    HostProcess DebugLaunch(const ProcessLaunchInfo &launch_info);

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

    DebugDelegateSP m_debug_delegate;

    HANDLE m_launched_event; // Signalled when the process is finished launching, either
                             // successfully or with an error.

    HostProcess m_process;    // The process being debugged.
    HostThread m_main_thread; // The main thread of the inferior.
    HANDLE m_image_file;      // The image file of the process being debugged.

    static lldb::thread_result_t DebuggerThreadRoutine(void *data);
    lldb::thread_result_t DebuggerThreadRoutine(const ProcessLaunchInfo &launch_info);
};
}

#endif
