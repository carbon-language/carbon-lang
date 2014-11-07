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
#include "LocalDebugDelegate.h"
#include "ProcessMessages.h"
#include "ProcessWindows.h"

using namespace lldb;
using namespace lldb_private;

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

ProcessWindows::ProcessWindows(Target& target, Listener &listener)
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
    HostProcess process;
    SetPrivateState(eStateLaunching);
    if (launch_info.GetFlags().Test(eLaunchFlagDebug))
    {
        DebugDelegateSP delegate(new LocalDebugDelegate(shared_from_this()));
        m_debugger.reset(new DebuggerThread(delegate));
        process = m_debugger->DebugLaunch(launch_info);
    }
    else
        return Host::LaunchProcess(launch_info);

    if (!result.Success())
        return result;

    launch_info.SetProcessID(process.GetProcessId());
    SetID(process.GetProcessId());
    return result;
}

Error
ProcessWindows::DoResume()
{
    Error error;
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
    error.SetErrorString("Detaching from processes is not currently supported on Windows.");
    return error;
}

Error
ProcessWindows::DoDestroy()
{
    Error error;
    error.SetErrorString("Destroying processes is not currently supported on Windows.");
    return error;
}

void
ProcessWindows::RefreshStateAfterStop()
{
}

bool
ProcessWindows::IsAlive()
{
    return false;
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
ProcessWindows::OnProcessLaunched(const ProcessMessageCreateProcess &message)
{
}

void
ProcessWindows::OnExitProcess(const ProcessMessageExitProcess &message)
{
    SetProcessExitStatus(nullptr, GetID(), true, 0, message.GetExitCode());
}

void
ProcessWindows::OnDebuggerConnected(const ProcessMessageDebuggerConnected &message)
{
}

void
ProcessWindows::OnDebugException(const ProcessMessageException &message)
{
}

void
ProcessWindows::OnCreateThread(const ProcessMessageCreateThread &message)
{
}

void
ProcessWindows::OnExitThread(const ProcessMessageExitThread &message)
{
}

void
ProcessWindows::OnLoadDll(const ProcessMessageLoadDll &message)
{
}

void
ProcessWindows::OnUnloadDll(const ProcessMessageUnloadDll &message)
{
}

void
ProcessWindows::OnDebugString(const ProcessMessageDebugString &message)
{
}

void
ProcessWindows::OnDebuggerError(const ProcessMessageDebuggerError &message)
{
}
