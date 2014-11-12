//===-- ProcessWindows.cpp --------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// Windows includes
#include "lldb/Host/windows/windows.h"

// C++ Includes
#include <vector>

// Other libraries and framework includes
#include "lldb/Core/Module.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Core/State.h"
#include "lldb/Host/Host.h"
#include "lldb/Host/HostProcess.h"
#include "lldb/Host/MonitoringProcessLauncher.h"
#include "lldb/Host/ThreadLauncher.h"
#include "lldb/Host/windows/ProcessLauncherWindows.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Target/DynamicLoader.h"
#include "lldb/Target/FileAction.h"
#include "lldb/Target/Target.h"

#include "DebuggerThread.h"
#include "ExceptionRecord.h"
#include "LocalDebugDelegate.h"
#include "ProcessWindows.h"

using namespace lldb;
using namespace lldb_private;

namespace lldb_private
{
// We store a pointer to this class in the ProcessWindows, so that we don't expose Windows
// OS specific types and implementation details from a public header file.
class ProcessWindowsData
{
  public:
    ProcessWindowsData(const ProcessLaunchInfo &launch_info)
        : m_initial_stop_event(nullptr)
        , m_launch_info(launch_info)
        , m_initial_stop_received(false)
    {
        m_initial_stop_event = ::CreateEvent(nullptr, TRUE, FALSE, nullptr);
    }

    ~ProcessWindowsData() { ::CloseHandle(m_initial_stop_event); }

    ProcessLaunchInfo m_launch_info;
    std::shared_ptr<lldb_private::ExceptionRecord> m_active_exception;
    lldb_private::Error m_launch_error;
    lldb_private::DebuggerThreadSP m_debugger;
    HANDLE m_initial_stop_event;
    bool m_initial_stop_received;
};
}
//------------------------------------------------------------------------------
// Static functions.

ProcessSP
ProcessWindows::CreateInstance(Target &target, Listener &listener, const FileSpec *)
{
    return ProcessSP(new ProcessWindows(target, listener));
}

void
ProcessWindows::Initialize()
{
    static bool g_initialized = false;

    if (!g_initialized)
    {
        g_initialized = true;
        PluginManager::RegisterPlugin(GetPluginNameStatic(),
                                      GetPluginDescriptionStatic(),
                                      CreateInstance);
    }
}

//------------------------------------------------------------------------------
// Constructors and destructors.

ProcessWindows::ProcessWindows(Target &target, Listener &listener)
    : lldb_private::Process(target, listener)
{
}

ProcessWindows::~ProcessWindows()
{
}

void
ProcessWindows::Terminate()
{
}

lldb_private::ConstString
ProcessWindows::GetPluginNameStatic()
{
    static ConstString g_name("windows");
    return g_name;
}

const char *
ProcessWindows::GetPluginDescriptionStatic()
{
    return "Process plugin for Windows";
}


bool
ProcessWindows::UpdateThreadList(ThreadList &old_thread_list, ThreadList &new_thread_list)
{
    new_thread_list = old_thread_list;
    return new_thread_list.GetSize(false) > 0;
}

Error
ProcessWindows::DoLaunch(Module *exe_module,
                         ProcessLaunchInfo &launch_info)
{
    Error result;
    if (!launch_info.GetFlags().Test(eLaunchFlagDebug))
    {
        result.SetErrorString("ProcessWindows can only be used to launch processes for debugging.");
        return result;
    }

    m_session_data.reset(new ProcessWindowsData(launch_info));

    SetPrivateState(eStateLaunching);
    DebugDelegateSP delegate(new LocalDebugDelegate(shared_from_this()));
    m_session_data->m_debugger.reset(new DebuggerThread(delegate));
    DebuggerThreadSP debugger = m_session_data->m_debugger;

    // Kick off the DebugLaunch asynchronously and wait for it to complete.
    result = debugger->DebugLaunch(launch_info);

    HostProcess process;
    if (result.Success())
    {
        if (::WaitForSingleObject(m_session_data->m_initial_stop_event, INFINITE) == WAIT_OBJECT_0)
            process = debugger->GetProcess();
        else
            result.SetError(::GetLastError(), eErrorTypeWin32);
    }

    if (!result.Success())
        return result;

    // We've hit the initial stop.  The private state should already be set to stopped as a result
    // of encountering the breakpoint exception.
    launch_info.SetProcessID(process.GetProcessId());
    SetID(process.GetProcessId());

    return result;
}

Error
ProcessWindows::DoResume()
{
    Error error;
    if (GetPrivateState() == eStateStopped)
    {
        if (m_session_data->m_active_exception)
        {
            // Resume the process and continue processing debug events.
            m_session_data->m_active_exception.reset();
            m_session_data->m_debugger->ContinueAsyncException(ExceptionResult::Handled);
        }

        SetPrivateState(eStateRunning);
    }
    return error;
}


//------------------------------------------------------------------------------
// ProcessInterface protocol.

lldb_private::ConstString
ProcessWindows::GetPluginName()
{
    return GetPluginNameStatic();
}

uint32_t
ProcessWindows::GetPluginVersion()
{
    return 1;
}

void
ProcessWindows::GetPluginCommandHelp(const char *command, Stream *strm)
{
}

Error
ProcessWindows::ExecutePluginCommand(Args &command, Stream *strm)
{
    return Error(1, eErrorTypeGeneric);
}

Log *
ProcessWindows::EnablePluginLogging(Stream *strm, Args &command)
{
    return NULL;
}

Error
ProcessWindows::DoDetach(bool keep_stopped)
{
    Error error;
    return error;
}

Error
ProcessWindows::DoDestroy()
{
    Error error;
    if (GetPrivateState() != eStateExited && GetPrivateState() != eStateDetached)
    {
        DebugActiveProcessStop(m_session_data->m_debugger->GetProcess().GetProcessId());
        SetPrivateState(eStateExited);
    }
    return error;
}

void
ProcessWindows::RefreshStateAfterStop()
{
}

bool
ProcessWindows::IsAlive()
{
    StateType state = GetPrivateState();
    switch (state)
    {
        case eStateCrashed:
        case eStateDetached:
        case eStateUnloaded:
        case eStateExited:
        case eStateInvalid:
            return false;
        default:
            return true;
    }
}

size_t
ProcessWindows::DoReadMemory(lldb::addr_t vm_addr,
                             void *buf,
                             size_t size,
                             Error &error)
{
    return 0;
}

bool
ProcessWindows::CanDebug(Target &target, bool plugin_specified_by_name)
{
    if (plugin_specified_by_name)
        return true;

    // For now we are just making sure the file exists for a given module
    ModuleSP exe_module_sp(target.GetExecutableModule());
    if (exe_module_sp.get())
        return exe_module_sp->GetFileSpec().Exists();
    return false;
}

void
ProcessWindows::OnExitProcess(uint32_t exit_code)
{
    SetProcessExitStatus(nullptr, GetID(), true, 0, exit_code);
    SetPrivateState(eStateExited);
}

void
ProcessWindows::OnDebuggerConnected(lldb::addr_t image_base)
{
    ModuleSP module = GetTarget().GetExecutableModule();
    bool load_addr_changed;
    module->SetLoadAddress(GetTarget(), image_base, false, load_addr_changed);
}

ExceptionResult
ProcessWindows::OnDebugException(bool first_chance, const ExceptionRecord &record)
{
    ExceptionResult result = ExceptionResult::NotHandled;
    m_session_data->m_active_exception.reset(new ExceptionRecord(record));
    switch (record.GetExceptionCode())
    {
        case EXCEPTION_BREAKPOINT:
            // Handle breakpoints at the first chance.
            result = ExceptionResult::WillHandle;

            if (!m_session_data->m_initial_stop_received)
            {
                m_session_data->m_initial_stop_received = true;
                ::SetEvent(m_session_data->m_initial_stop_event);
            }
            break;
        default:
            // For non-breakpoints, give the application a chance to handle the exception first.
            if (first_chance)
                result = ExceptionResult::NotHandled;
            else
                result = ExceptionResult::WillHandle;
    }

    if (!first_chance)
    {
        // Any second chance exception is an application crash by definition.
        SetPrivateState(eStateCrashed);
    }
    else if (result == ExceptionResult::WillHandle)
    {
        // For first chance exceptions that we can handle, the process is stopped so the user
        // can inspect / manipulate the state of the process in the debugger.
        SetPrivateState(eStateStopped);
    }
    else
    {
        // For first chance exceptions that we either eat or send back to the application, don't
        // modify the state of the application.
    }

    return result;
}

void
ProcessWindows::OnCreateThread(const HostThread &thread)
{
}

void
ProcessWindows::OnExitThread(const HostThread &thread)
{
}

void
ProcessWindows::OnLoadDll(const ModuleSpec &module_spec, lldb::addr_t module_addr)
{
    // Confusingly, there is no Target::AddSharedModule.  Instead, calling GetSharedModule() with
    // a new module will add it to the module list and return a corresponding ModuleSP.
    Error error;
    ModuleSP module = GetTarget().GetSharedModule(module_spec, &error);
    bool load_addr_changed = false;
    module->SetLoadAddress(GetTarget(), module_addr, false, load_addr_changed);
}

void
ProcessWindows::OnUnloadDll(lldb::addr_t module_addr)
{
    // TODO: Figure out how to get the ModuleSP loaded at the specified address and remove
    // it from the target's module list.
}

void
ProcessWindows::OnDebugString(const std::string &string)
{
}

void
ProcessWindows::OnDebuggerError(const Error &error, uint32_t type)
{
    if (!m_session_data->m_initial_stop_received)
    {
        // If we haven't actually launched the process yet, this was an error
        // launching the process.  Set the internal error and signal.
        m_session_data->m_launch_error = error;
        ::SetEvent(m_session_data->m_initial_stop_event);
        return;
    }

    // This happened while debugging.  Do we shutdown the debugging session, try to continue,
    // or do something else?
}
