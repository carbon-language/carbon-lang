//===-- DebugDriverThread.cpp -----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "DebugDriverThread.h"
#include "DebugOneProcessThread.h"
#include "DriverMessages.h"
#include "DriverMessageResults.h"
#include "ProcessMessages.h"

#include "lldb/Core/Error.h"
#include "lldb/Core/Log.h"
#include "lldb/Host/Predicate.h"
#include "lldb/Host/ThisThread.h"
#include "lldb/Host/ThreadLauncher.h"
#include "lldb/Host/windows/HostProcessWindows.h"
#include "lldb/Host/windows/HostThreadWindows.h"
#include "lldb/Host/windows/ProcessLauncherWindows.h"

#include "llvm/Support/raw_ostream.h"

using namespace lldb;
using namespace lldb_private;

namespace
{
struct DebugLaunchContext
{
    DebugOneProcessThread *instance;
    const DriverLaunchProcessMessage *launch;
};
}

DebugOneProcessThread::DebugOneProcessThread(HostThread driver_thread)
    : m_driver_thread(driver_thread)
    , m_pending_create(nullptr)
    , m_image_file(nullptr)
{
    m_launch_predicate.SetValue(nullptr, eBroadcastNever);
}

DebugOneProcessThread::~DebugOneProcessThread()
{
}

const DriverLaunchProcessMessageResult *
DebugOneProcessThread::DebugLaunch(const DriverLaunchProcessMessage *message)
{
    Error error;
    const DriverLaunchProcessMessageResult *result = nullptr;
    DebugLaunchContext context;
    context.instance = this;
    context.launch = message;
    HostThread slave_thread(ThreadLauncher::LaunchThread("lldb.plugin.process-windows.slave[?]", DebugLaunchThread, &context, &error));
    if (error.Success())
        m_launch_predicate.WaitForValueNotEqualTo(nullptr, result);

    return result;
}

lldb::thread_result_t
DebugOneProcessThread::DebugLaunchThread(void *data)
{
    DebugLaunchContext *context = static_cast<DebugLaunchContext *>(data);
    DebugOneProcessThread *thread = context->instance;
    return thread->DebugLaunchThread(context->launch);
}

lldb::thread_result_t
DebugOneProcessThread::DebugLaunchThread(const DriverLaunchProcessMessage *message)
{
    // Grab a shared_ptr reference to this so that we know it won't get deleted until after the
    // thread routine has exited.
    std::shared_ptr<DebugOneProcessThread> this_ref(shared_from_this());
    Log *log(lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_PROCESS));

    Error error;
    ProcessLauncherWindows launcher;
    HostProcess process(launcher.LaunchProcess(message->GetLaunchInfo(), error));
    // If we couldn't create the process, return the result immediately.  Otherwise enter the debug
    // loop and wait until we get the create process debug notification.  Note that if the process
    // was created successfully, we can throw away the process handle we got from CreateProcess
    // because Windows will give us another (potentially more useful?) handle when it sends us the
    // CREATE_PROCESS_DEBUG_EVENT.
    if (error.Success())
    {
        m_pending_create = message;
        m_pending_create->Retain();
        DebugLoop();
    }
    else
    {
        DriverLaunchProcessMessageResult *result = DriverLaunchProcessMessageResult::Create(m_pending_create);
        result->SetError(error);
        result->SetProcess(process);
        m_launch_predicate.SetValue(result, eBroadcastAlways);
    }

    return 0;
}

void
DebugOneProcessThread::DebugLoop()
{
    DEBUG_EVENT dbe = {0};
    bool exit = false;
    while (!exit && WaitForDebugEvent(&dbe, INFINITE))
    {
        DWORD continue_status = DBG_CONTINUE;
        switch (dbe.dwDebugEventCode)
        {
            case EXCEPTION_DEBUG_EVENT:
                continue_status = HandleExceptionEvent(dbe.u.Exception, dbe.dwThreadId);
                break;
            case CREATE_THREAD_DEBUG_EVENT:
                continue_status = HandleCreateThreadEvent(dbe.u.CreateThread, dbe.dwThreadId);
                break;
            case CREATE_PROCESS_DEBUG_EVENT:
                continue_status = HandleCreateProcessEvent(dbe.u.CreateProcessInfo, dbe.dwThreadId);
                break;
            case EXIT_THREAD_DEBUG_EVENT:
                continue_status = HandleExitThreadEvent(dbe.u.ExitThread, dbe.dwThreadId);
                break;
            case EXIT_PROCESS_DEBUG_EVENT:
                continue_status = HandleExitProcessEvent(dbe.u.ExitProcess, dbe.dwThreadId);
                exit = true;
                break;
            case LOAD_DLL_DEBUG_EVENT:
                continue_status = HandleLoadDllEvent(dbe.u.LoadDll, dbe.dwThreadId);
                break;
            case UNLOAD_DLL_DEBUG_EVENT:
                continue_status = HandleUnloadDllEvent(dbe.u.UnloadDll, dbe.dwThreadId);
                break;
            case OUTPUT_DEBUG_STRING_EVENT:
                continue_status = HandleODSEvent(dbe.u.DebugString, dbe.dwThreadId);
                break;
            case RIP_EVENT:
                continue_status = HandleRipEvent(dbe.u.RipInfo, dbe.dwThreadId);
                if (dbe.u.RipInfo.dwType == SLE_ERROR)
                    exit = true;
                break;
        }

        ::ContinueDebugEvent(dbe.dwProcessId, dbe.dwThreadId, continue_status);
    }
}

DWORD
DebugOneProcessThread::HandleExceptionEvent(const EXCEPTION_DEBUG_INFO &info, DWORD thread_id)
{
    return DBG_CONTINUE;
}

DWORD
DebugOneProcessThread::HandleCreateThreadEvent(const CREATE_THREAD_DEBUG_INFO &info, DWORD thread_id)
{
    return DBG_CONTINUE;
}

DWORD
DebugOneProcessThread::HandleCreateProcessEvent(const CREATE_PROCESS_DEBUG_INFO &info, DWORD thread_id)
{
    std::string thread_name;
    llvm::raw_string_ostream name_stream(thread_name);
    name_stream << "lldb.plugin.process-windows.slave[" << m_process.GetProcessId() << "]";
    name_stream.flush();
    ThisThread::SetName(thread_name.c_str());

    // info.hProcess and info.hThread are closed automatically by Windows when
    // EXIT_PROCESS_DEBUG_EVENT is received.
    m_process = HostProcess(info.hProcess);
    ((HostProcessWindows &)m_process.GetNativeProcess()).SetOwnsHandle(false);
    m_main_thread = HostThread(info.hThread);
    ((HostThreadWindows &)m_main_thread.GetNativeThread()).SetOwnsHandle(false);
    m_image_file = info.hFile;

    DriverLaunchProcessMessageResult *result = DriverLaunchProcessMessageResult::Create(m_pending_create);
    result->SetError(Error());
    result->SetProcess(m_process);
    m_launch_predicate.SetValue(result, eBroadcastAlways);

    m_pending_create->Release();
    m_pending_create = nullptr;

    return DBG_CONTINUE;
}

DWORD
DebugOneProcessThread::HandleExitThreadEvent(const EXIT_THREAD_DEBUG_INFO &info, DWORD thread_id)
{
    return DBG_CONTINUE;
}

DWORD
DebugOneProcessThread::HandleExitProcessEvent(const EXIT_PROCESS_DEBUG_INFO &info, DWORD thread_id)
{
    HANDLE driver = m_driver_thread.GetNativeThread().GetSystemHandle();
    ProcessMessageExitProcess *message = new ProcessMessageExitProcess(m_process, info.dwExitCode);

    QueueUserAPC(NotifySlaveProcessExited, driver, reinterpret_cast<ULONG_PTR>(message));

    m_process = HostProcess();
    m_main_thread = HostThread();
    ::CloseHandle(m_image_file);
    m_image_file = nullptr;
    return DBG_CONTINUE;
}

DWORD
DebugOneProcessThread::HandleLoadDllEvent(const LOAD_DLL_DEBUG_INFO &info, DWORD thread_id)
{
    // Windows does not automatically close info.hFile when the DLL is unloaded.
    ::CloseHandle(info.hFile);
    return DBG_CONTINUE;
}

DWORD
DebugOneProcessThread::HandleUnloadDllEvent(const UNLOAD_DLL_DEBUG_INFO &info, DWORD thread_id)
{
    return DBG_CONTINUE;
}

DWORD
DebugOneProcessThread::HandleODSEvent(const OUTPUT_DEBUG_STRING_INFO &info, DWORD thread_id)
{
    return DBG_CONTINUE;
}

DWORD
DebugOneProcessThread::HandleRipEvent(const RIP_INFO &info, DWORD thread_id)
{
    HANDLE driver = m_driver_thread.GetNativeThread().GetSystemHandle();
    Error error(info.dwError, eErrorTypeWin32);
    ProcessMessageDebuggerError *message = new ProcessMessageDebuggerError(m_process, error, info.dwType);

    QueueUserAPC(NotifySlaveRipEvent, driver, reinterpret_cast<ULONG_PTR>(message));

    return DBG_CONTINUE;
}

void
DebugOneProcessThread::NotifySlaveProcessExited(ULONG_PTR message)
{
    ProcessMessageExitProcess *slave_message = reinterpret_cast<ProcessMessageExitProcess *>(message);
    DebugDriverThread::GetInstance().OnExitProcess(*slave_message);
    delete slave_message;
}

void
DebugOneProcessThread::NotifySlaveRipEvent(ULONG_PTR message)
{
    ProcessMessageDebuggerError *slave_message = reinterpret_cast<ProcessMessageDebuggerError *>(message);
    DebugDriverThread::GetInstance().OnDebuggerError(*slave_message);
    delete slave_message;
}
