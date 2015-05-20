//===-- DebuggerThread.DebuggerThread --------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "DebuggerThread.h"
#include "ExceptionRecord.h"
#include "IDebugDelegate.h"

#include "lldb/Core/Error.h"
#include "lldb/Core/Log.h"
#include "lldb/Core/ModuleSpec.h"
#include "lldb/Host/FileSpec.h"
#include "lldb/Host/Predicate.h"
#include "lldb/Host/ThisThread.h"
#include "lldb/Host/ThreadLauncher.h"
#include "lldb/Host/windows/HostProcessWindows.h"
#include "lldb/Host/windows/HostThreadWindows.h"
#include "lldb/Host/windows/ProcessLauncherWindows.h"
#include "lldb/Target/ProcessLaunchInfo.h"
#include "lldb/Target/Process.h"

#include "Plugins/Process/Windows/ProcessWindowsLog.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

using namespace lldb;
using namespace lldb_private;

namespace
{
struct DebugLaunchContext
{
    DebugLaunchContext(DebuggerThread *thread, const ProcessLaunchInfo &launch_info)
        : m_thread(thread)
        , m_launch_info(launch_info)
    {
    }
    DebuggerThread *m_thread;
    ProcessLaunchInfo m_launch_info;
};

struct DebugAttachContext
{
    DebugAttachContext(DebuggerThread *thread, lldb::pid_t pid, const ProcessAttachInfo &attach_info)
        : m_thread(thread)
        , m_pid(pid)
        , m_attach_info(attach_info)
    {
    }
    DebuggerThread *m_thread;
    lldb::pid_t m_pid;
    ProcessAttachInfo m_attach_info;
};
}

DebuggerThread::DebuggerThread(DebugDelegateSP debug_delegate)
    : m_debug_delegate(debug_delegate)
    , m_image_file(nullptr)
{
}

DebuggerThread::~DebuggerThread()
{
}

Error
DebuggerThread::DebugLaunch(const ProcessLaunchInfo &launch_info)
{
    WINLOG_IFALL(WINDOWS_LOG_PROCESS,
        "DebuggerThread::DebugLaunch launching '%s'", launch_info.GetExecutableFile().GetPath().c_str());

    Error error;
    DebugLaunchContext *context = new DebugLaunchContext(this, launch_info);
    HostThread slave_thread(ThreadLauncher::LaunchThread("lldb.plugin.process-windows.slave[?]",
                                                         DebuggerThreadLaunchRoutine, context, &error));

    if (!error.Success())
    {
        WINERR_IFALL(WINDOWS_LOG_PROCESS,
            "DebugLaunch couldn't launch debugger thread.  %s", error.AsCString());
    }

    return error;
}

Error
DebuggerThread::DebugAttach(lldb::pid_t pid, const ProcessAttachInfo &attach_info)
{
    WINLOG_IFALL(WINDOWS_LOG_PROCESS, "DebuggerThread::DebugAttach attaching to '%u'", (DWORD)pid);

    Error error;
    DebugAttachContext *context = new DebugAttachContext(this, pid, attach_info);
    HostThread slave_thread(ThreadLauncher::LaunchThread("lldb.plugin.process-windows.slave[?]",
                                                         DebuggerThreadAttachRoutine, context, &error));

    if (!error.Success())
    {
        WINERR_IFALL(WINDOWS_LOG_PROCESS, "DebugAttach couldn't attach to process '%u'.  %s", (DWORD)pid,
                     error.AsCString());
    }

    return error;
}

lldb::thread_result_t
DebuggerThread::DebuggerThreadLaunchRoutine(void *data)
{
    DebugLaunchContext *context = static_cast<DebugLaunchContext *>(data);
    lldb::thread_result_t result = context->m_thread->DebuggerThreadLaunchRoutine(context->m_launch_info);
    delete context;
    return result;
}

lldb::thread_result_t
DebuggerThread::DebuggerThreadAttachRoutine(void *data)
{
    DebugAttachContext *context = static_cast<DebugAttachContext *>(data);
    lldb::thread_result_t result =
        context->m_thread->DebuggerThreadAttachRoutine(context->m_pid, context->m_attach_info);
    delete context;
    return result;
}

lldb::thread_result_t
DebuggerThread::DebuggerThreadLaunchRoutine(const ProcessLaunchInfo &launch_info)
{
    // Grab a shared_ptr reference to this so that we know it won't get deleted until after the
    // thread routine has exited.
    std::shared_ptr<DebuggerThread> this_ref(shared_from_this());

    WINLOG_IFALL(WINDOWS_LOG_PROCESS, "DebuggerThread preparing to launch '%s' on background thread.",
                 launch_info.GetExecutableFile().GetPath().c_str());

    Error error;
    ProcessLauncherWindows launcher;
    HostProcess process(launcher.LaunchProcess(launch_info, error));
    // If we couldn't create the process, notify waiters immediately.  Otherwise enter the debug
    // loop and wait until we get the create process debug notification.  Note that if the process
    // was created successfully, we can throw away the process handle we got from CreateProcess
    // because Windows will give us another (potentially more useful?) handle when it sends us the
    // CREATE_PROCESS_DEBUG_EVENT.
    if (error.Success())
        DebugLoop();
    else
        m_debug_delegate->OnDebuggerError(error, 0);

    return 0;
}

lldb::thread_result_t
DebuggerThread::DebuggerThreadAttachRoutine(lldb::pid_t pid, const ProcessAttachInfo &attach_info)
{
    // Grab a shared_ptr reference to this so that we know it won't get deleted until after the
    // thread routine has exited.
    std::shared_ptr<DebuggerThread> this_ref(shared_from_this());

    WINLOG_IFALL(WINDOWS_LOG_PROCESS, "DebuggerThread preparing to attach to process '%u' on background thread.",
                 (DWORD)pid);

    if (!DebugActiveProcess((DWORD)pid))
    {
        Error error(::GetLastError(), eErrorTypeWin32);
        m_debug_delegate->OnDebuggerError(error, 0);
        return 0;
    }

    // The attach was successful, enter the debug loop.  From here on out, this is no different than
    // a create process operation, so all the same comments in DebugLaunch should apply from this
    // point out.
    DebugLoop();

    return 0;
}

Error
DebuggerThread::StopDebugging(bool terminate)
{
    Error error;

    lldb::pid_t pid = m_process.GetProcessId();

    WINLOG_IFALL(WINDOWS_LOG_PROCESS,
        "StopDebugging('%s') called (inferior=%I64u).",
        (terminate ? "true" : "false"), pid);

    if (terminate)
    {
        // Make a copy of the process, since the termination sequence will reset
        // DebuggerThread's internal copy and it needs to remain open for the Wait operation.
        HostProcess process_copy = m_process;
        lldb::process_t handle = m_process.GetNativeProcess().GetSystemHandle();

        // Initiate the termination before continuing the exception, so that the next debug
        // event we get is the exit process event, and not some other event.
        BOOL terminate_suceeded = TerminateProcess(handle, 0);
        WINLOG_IFALL(WINDOWS_LOG_PROCESS,
            "StopDebugging called TerminateProcess(0x%p, 0) (inferior=%I64u), success='%s'",
            handle, pid, (terminate_suceeded ? "true" : "false"));


        // If we're stuck waiting for an exception to continue, continue it now.  But only
        // AFTER setting the termination event, to make sure that we don't race and enter
        // another wait for another debug event.
        if (m_active_exception.get())
        {
            WINLOG_IFANY(WINDOWS_LOG_PROCESS|WINDOWS_LOG_EXCEPTION,
                "StopDebugging masking active exception");

            ContinueAsyncException(ExceptionResult::MaskException);
        }

        // Don't return until the process has exited.
        if (terminate_suceeded)
        {
            WINLOG_IFALL(WINDOWS_LOG_PROCESS,
                "StopDebugging waiting for termination of process %u to complete.", pid);

            DWORD wait_result = ::WaitForSingleObject(handle, 5000);
            if (wait_result != WAIT_OBJECT_0)
                terminate_suceeded = false;

            WINLOG_IFALL(WINDOWS_LOG_PROCESS,
                "StopDebugging WaitForSingleObject(0x%p, 5000) returned %u",
                handle, wait_result);
        }

        if (!terminate_suceeded)
            error.SetError(GetLastError(), eErrorTypeWin32);
    }
    else
    {
        error.SetErrorString("Detach not yet supported on Windows.");
        // TODO: Implement detach.
    }

    if (!error.Success())
    {
        WINERR_IFALL(WINDOWS_LOG_PROCESS,
            "StopDebugging encountered an error while trying to  stop process %u.  %s",
            pid, error.AsCString());
    }
    return error;
}

void
DebuggerThread::ContinueAsyncException(ExceptionResult result)
{
    if (!m_active_exception.get())
        return;

    WINLOG_IFANY(WINDOWS_LOG_PROCESS|WINDOWS_LOG_EXCEPTION,
        "ContinueAsyncException called for inferior process %I64u, broadcasting.",
        m_process.GetProcessId());

    m_active_exception.reset();
    m_exception_pred.SetValue(result, eBroadcastAlways);
}

void
DebuggerThread::FreeProcessHandles()
{
    m_process = HostProcess();
    m_main_thread = HostThread();
    if (m_image_file)
    {
        ::CloseHandle(m_image_file);
        m_image_file = nullptr;
    }
}

void
DebuggerThread::DebugLoop()
{
    DEBUG_EVENT dbe = {0};
    bool should_debug = true;
    WINLOG_IFALL(WINDOWS_LOG_EVENT, "Entering WaitForDebugEvent loop");
    while (should_debug)
    {
        WINLOGD_IFALL(WINDOWS_LOG_EVENT, "Calling WaitForDebugEvent");
        BOOL wait_result = WaitForDebugEvent(&dbe, INFINITE);
        if (wait_result)
        {
            DWORD continue_status = DBG_CONTINUE;
            switch (dbe.dwDebugEventCode)
            {
                case EXCEPTION_DEBUG_EVENT:
                {
                    ExceptionResult status = HandleExceptionEvent(dbe.u.Exception, dbe.dwThreadId);

                    if (status == ExceptionResult::MaskException)
                        continue_status = DBG_CONTINUE;
                    else if (status == ExceptionResult::SendToApplication)
                        continue_status = DBG_EXCEPTION_NOT_HANDLED;
                    break;
                }
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
                    should_debug = false;
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
                        should_debug = false;
                    break;
            }

            WINLOGD_IFALL(WINDOWS_LOG_EVENT, "DebugLoop calling ContinueDebugEvent(%u, %u, %u) on thread %u.",
                          dbe.dwProcessId, dbe.dwThreadId, continue_status, ::GetCurrentThreadId());

            ::ContinueDebugEvent(dbe.dwProcessId, dbe.dwThreadId, continue_status);
        }
        else
        {
            WINERR_IFALL(WINDOWS_LOG_EVENT,
                "DebugLoop returned FALSE from WaitForDebugEvent.  Error = %u",
                ::GetCurrentThreadId(), ::GetLastError());

            should_debug = false;
        }
    }
    FreeProcessHandles();
}

ExceptionResult
DebuggerThread::HandleExceptionEvent(const EXCEPTION_DEBUG_INFO &info, DWORD thread_id)
{
    bool first_chance = (info.dwFirstChance != 0);

    m_active_exception.reset(new ExceptionRecord(info.ExceptionRecord, thread_id));
    WINLOG_IFANY(WINDOWS_LOG_EVENT | WINDOWS_LOG_EXCEPTION,
                 "HandleExceptionEvent encountered %s chance exception 0x%x on thread 0x%x",
                 first_chance ? "first" : "second", info.ExceptionRecord.ExceptionCode, thread_id);

    ExceptionResult result = m_debug_delegate->OnDebugException(first_chance,
                                                                *m_active_exception);
    m_exception_pred.SetValue(result, eBroadcastNever);

    WINLOG_IFANY(WINDOWS_LOG_EVENT|WINDOWS_LOG_EXCEPTION,
        "DebuggerThread::HandleExceptionEvent waiting for ExceptionPred != BreakInDebugger");

    m_exception_pred.WaitForValueNotEqualTo(ExceptionResult::BreakInDebugger, result);

    WINLOG_IFANY(WINDOWS_LOG_EVENT|WINDOWS_LOG_EXCEPTION,
        "DebuggerThread::HandleExceptionEvent got ExceptionPred = %u",
         m_exception_pred.GetValue());

    return result;
}

DWORD
DebuggerThread::HandleCreateThreadEvent(const CREATE_THREAD_DEBUG_INFO &info, DWORD thread_id)
{
    WINLOG_IFANY(WINDOWS_LOG_EVENT|WINDOWS_LOG_THREAD,
        "HandleCreateThreadEvent Thread 0x%x spawned in process %I64u",
        thread_id, m_process.GetProcessId());
    HostThread thread(info.hThread);
    thread.GetNativeThread().SetOwnsHandle(false);
    m_debug_delegate->OnCreateThread(thread);
    return DBG_CONTINUE;
}

DWORD
DebuggerThread::HandleCreateProcessEvent(const CREATE_PROCESS_DEBUG_INFO &info, DWORD thread_id)
{
    uint32_t process_id = ::GetProcessId(info.hProcess);

    WINLOG_IFANY(WINDOWS_LOG_EVENT | WINDOWS_LOG_PROCESS, "HandleCreateProcessEvent process %u spawned", process_id);

    std::string thread_name;
    llvm::raw_string_ostream name_stream(thread_name);
    name_stream << "lldb.plugin.process-windows.slave[" << process_id << "]";
    name_stream.flush();
    ThisThread::SetName(thread_name.c_str());

    // info.hProcess and info.hThread are closed automatically by Windows when
    // EXIT_PROCESS_DEBUG_EVENT is received.
    m_process = HostProcess(info.hProcess);
    ((HostProcessWindows &)m_process.GetNativeProcess()).SetOwnsHandle(false);
    m_main_thread = HostThread(info.hThread);
    m_main_thread.GetNativeThread().SetOwnsHandle(false);
    m_image_file = info.hFile;

    lldb::addr_t load_addr = reinterpret_cast<lldb::addr_t>(info.lpBaseOfImage);
    m_debug_delegate->OnDebuggerConnected(load_addr);

    return DBG_CONTINUE;
}

DWORD
DebuggerThread::HandleExitThreadEvent(const EXIT_THREAD_DEBUG_INFO &info, DWORD thread_id)
{
    WINLOG_IFANY(WINDOWS_LOG_EVENT|WINDOWS_LOG_THREAD,
        "HandleExitThreadEvent Thread %u exited with code %u in process %I64u",
        thread_id, info.dwExitCode, m_process.GetProcessId());
    m_debug_delegate->OnExitThread(thread_id, info.dwExitCode);
    return DBG_CONTINUE;
}

DWORD
DebuggerThread::HandleExitProcessEvent(const EXIT_PROCESS_DEBUG_INFO &info, DWORD thread_id)
{
    WINLOG_IFANY(WINDOWS_LOG_EVENT|WINDOWS_LOG_THREAD,
        "HandleExitProcessEvent process %I64u exited with code %u",
        m_process.GetProcessId(), info.dwExitCode);

    m_debug_delegate->OnExitProcess(info.dwExitCode);

    FreeProcessHandles();
    return DBG_CONTINUE;
}

DWORD
DebuggerThread::HandleLoadDllEvent(const LOAD_DLL_DEBUG_INFO &info, DWORD thread_id)
{
    if (info.hFile == nullptr)
    {
        // Not sure what this is, so just ignore it.
        WINWARN_IFALL(WINDOWS_LOG_EVENT, "Inferior %I64u - HandleLoadDllEvent has a NULL file handle, returning...",
                      m_process.GetProcessId());
        return DBG_CONTINUE;
    }

    std::vector<char> buffer(1);
    DWORD required_size = GetFinalPathNameByHandle(info.hFile, &buffer[0], 0, VOLUME_NAME_DOS);
    if (required_size > 0)
    {
        buffer.resize(required_size + 1);
        required_size = GetFinalPathNameByHandle(info.hFile, &buffer[0], required_size + 1, VOLUME_NAME_DOS);
        llvm::StringRef path_str(&buffer[0]);
        const char *path = path_str.data();
        if (path_str.startswith("\\\\?\\"))
            path += 4;

        FileSpec file_spec(path, false);
        ModuleSpec module_spec(file_spec);
        lldb::addr_t load_addr = reinterpret_cast<lldb::addr_t>(info.lpBaseOfDll);

        WINLOG_IFALL(WINDOWS_LOG_EVENT, "Inferior %I64u - HandleLoadDllEvent DLL '%s' loaded at address 0x%p...",
                     m_process.GetProcessId(), path, info.lpBaseOfDll);

        m_debug_delegate->OnLoadDll(module_spec, load_addr);
    }
    else
    {
        WINERR_IFALL(WINDOWS_LOG_EVENT,
                     "Inferior %I64u - HandleLoadDllEvent Error %u occurred calling GetFinalPathNameByHandle",
                     m_process.GetProcessId(), ::GetLastError());
    }
    // Windows does not automatically close info.hFile, so we need to do it.
    ::CloseHandle(info.hFile);
    return DBG_CONTINUE;
}

DWORD
DebuggerThread::HandleUnloadDllEvent(const UNLOAD_DLL_DEBUG_INFO &info, DWORD thread_id)
{
    WINLOG_IFALL(WINDOWS_LOG_EVENT,
        "HandleUnloadDllEvent process %I64u unloading DLL at addr 0x%p.",
        m_process.GetProcessId(), info.lpBaseOfDll);

    m_debug_delegate->OnUnloadDll(reinterpret_cast<lldb::addr_t>(info.lpBaseOfDll));
    return DBG_CONTINUE;
}

DWORD
DebuggerThread::HandleODSEvent(const OUTPUT_DEBUG_STRING_INFO &info, DWORD thread_id)
{
    return DBG_CONTINUE;
}

DWORD
DebuggerThread::HandleRipEvent(const RIP_INFO &info, DWORD thread_id)
{
    WINERR_IFALL(WINDOWS_LOG_EVENT,
        "HandleRipEvent encountered error %u (type=%u) in process %I64u thread %u",
        info.dwError, info.dwType, m_process.GetProcessId(), thread_id);

    Error error(info.dwError, eErrorTypeWin32);
    m_debug_delegate->OnDebuggerError(error, info.dwType);

    return DBG_CONTINUE;
}
