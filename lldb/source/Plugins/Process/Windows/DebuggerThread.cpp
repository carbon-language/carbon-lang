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
    Error error;

    DebugLaunchContext *context = new DebugLaunchContext(this, launch_info);
    HostThread slave_thread(ThreadLauncher::LaunchThread("lldb.plugin.process-windows.slave[?]", DebuggerThreadRoutine, context, &error));

    return error;
}

lldb::thread_result_t
DebuggerThread::DebuggerThreadRoutine(void *data)
{
    DebugLaunchContext *context = static_cast<DebugLaunchContext *>(data);
    lldb::thread_result_t result = context->m_thread->DebuggerThreadRoutine(context->m_launch_info);
    delete context;
    return result;
}

lldb::thread_result_t
DebuggerThread::DebuggerThreadRoutine(const ProcessLaunchInfo &launch_info)
{
    // Grab a shared_ptr reference to this so that we know it won't get deleted until after the
    // thread routine has exited.
    std::shared_ptr<DebuggerThread> this_ref(shared_from_this());
    Log *log(lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_PROCESS));

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

Error
DebuggerThread::StopDebugging(bool terminate)
{
    Error error;

    if (terminate)
    {
        // Make a copy of the process, since the termination sequence will reset DebuggerThread's
        // internal copy and it needs to remain open for us to perform the Wait operation.
        HostProcess process_copy = m_process;
        lldb::process_t handle = process_copy.GetNativeProcess().GetSystemHandle();

        // Initiate the termination before continuing the exception, so that the next debug event
        // we get is the exit process event, and not some other event.
        BOOL terminate_suceeded = TerminateProcess(handle, 0);

        // If we're stuck waiting for an exception to continue, continue it now.  But only
        // AFTER setting the termination event, to make sure that we don't race and enter
        // another wait for another debug event.
        if (m_active_exception.get())
            ContinueAsyncException(ExceptionResult::MaskException);

        // Don't return until the process has exited.
        if (terminate_suceeded)
        {
            DWORD wait_result = ::WaitForSingleObject(handle, 5000);
            if (wait_result != WAIT_OBJECT_0)
                terminate_suceeded = false;
        }

        if (!terminate_suceeded)
            error.SetError(GetLastError(), eErrorTypeWin32);
    }
    else
    {
        error.SetErrorString("Detach not yet supported on Windows.");
        // TODO: Implement detach.
    }
    return error;
}

void
DebuggerThread::ContinueAsyncException(ExceptionResult result)
{
    if (!m_active_exception.get())
        return;

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
    while (should_debug && WaitForDebugEvent(&dbe, INFINITE))
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

        ::ContinueDebugEvent(dbe.dwProcessId, dbe.dwThreadId, continue_status);
    }
    FreeProcessHandles();
}

ExceptionResult
DebuggerThread::HandleExceptionEvent(const EXCEPTION_DEBUG_INFO &info, DWORD thread_id)
{
    bool first_chance = (info.dwFirstChance != 0);

    m_active_exception.reset(new ExceptionRecord(info.ExceptionRecord));

    ExceptionResult result = m_debug_delegate->OnDebugException(first_chance, *m_active_exception);
    m_exception_pred.SetValue(result, eBroadcastNever);
    m_exception_pred.WaitForValueNotEqualTo(ExceptionResult::BreakInDebugger, result);

    return result;
}

DWORD
DebuggerThread::HandleCreateThreadEvent(const CREATE_THREAD_DEBUG_INFO &info, DWORD thread_id)
{
    return DBG_CONTINUE;
}

DWORD
DebuggerThread::HandleCreateProcessEvent(const CREATE_PROCESS_DEBUG_INFO &info, DWORD thread_id)
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
    m_main_thread.GetNativeThread().SetOwnsHandle(false);
    m_image_file = info.hFile;

    lldb::addr_t load_addr = reinterpret_cast<lldb::addr_t>(info.lpBaseOfImage);
    m_debug_delegate->OnDebuggerConnected(load_addr);

    return DBG_CONTINUE;
}

DWORD
DebuggerThread::HandleExitThreadEvent(const EXIT_THREAD_DEBUG_INFO &info, DWORD thread_id)
{
    return DBG_CONTINUE;
}

DWORD
DebuggerThread::HandleExitProcessEvent(const EXIT_PROCESS_DEBUG_INFO &info, DWORD thread_id)
{
    FreeProcessHandles();

    m_debug_delegate->OnExitProcess(info.dwExitCode);
    return DBG_CONTINUE;
}

DWORD
DebuggerThread::HandleLoadDllEvent(const LOAD_DLL_DEBUG_INFO &info, DWORD thread_id)
{
    if (info.hFile == nullptr)
    {
        // Not sure what this is, so just ignore it.
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
        m_debug_delegate->OnLoadDll(module_spec, load_addr);
    }
    else
    {
        // An unknown error occurred getting the path name.
    }
    // Windows does not automatically close info.hFile, so we need to do it.
    ::CloseHandle(info.hFile);
    return DBG_CONTINUE;
}

DWORD
DebuggerThread::HandleUnloadDllEvent(const UNLOAD_DLL_DEBUG_INFO &info, DWORD thread_id)
{
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
    Error error(info.dwError, eErrorTypeWin32);
    m_debug_delegate->OnDebuggerError(error, info.dwType);

    return DBG_CONTINUE;
}
