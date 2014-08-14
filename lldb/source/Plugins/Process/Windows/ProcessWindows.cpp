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
// Other libraries and framework includes
#include "lldb/Core/Module.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Core/State.h"
#include "lldb/Host/Host.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Target/DynamicLoader.h"
#include "lldb/Target/FileAction.h"
#include "lldb/Target/Target.h"

#include "ProcessWindows.h"

using namespace lldb;
using namespace lldb_private;

namespace
{
HANDLE
GetStdioHandle(ProcessLaunchInfo &launch_info, int fd)
{
    const FileAction *action = launch_info.GetFileActionForFD(fd);
    if (action == nullptr)
        return NULL;
    SECURITY_ATTRIBUTES secattr = {0};
    secattr.nLength = sizeof(SECURITY_ATTRIBUTES);
    secattr.bInheritHandle = TRUE;

    const char *path = action->GetPath();
    DWORD access = 0;
    DWORD share = FILE_SHARE_READ | FILE_SHARE_WRITE;
    DWORD create = 0;
    DWORD flags = 0;
    if (fd == STDIN_FILENO)
    {
        access = GENERIC_READ;
        create = OPEN_EXISTING;
        flags = FILE_ATTRIBUTE_READONLY;
    }
    if (fd == STDOUT_FILENO || fd == STDERR_FILENO)
    {
        access = GENERIC_WRITE;
        create = CREATE_ALWAYS;
        if (fd == STDERR_FILENO)
            flags = FILE_FLAG_WRITE_THROUGH;
    }

    HANDLE result = ::CreateFile(path, access, share, &secattr, create, flags, NULL);
    return (result == INVALID_HANDLE_VALUE) ? NULL : result;
}
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

ProcessWindows::ProcessWindows(Target& target, Listener &listener)
    : lldb_private::Process(target, listener)
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
    std::string executable;
    std::string commandLine;
    std::vector<char> environment;
    STARTUPINFO startupinfo = {0};
    PROCESS_INFORMATION pi = {0};

    HANDLE stdin_handle = GetStdioHandle(launch_info, STDIN_FILENO);
    HANDLE stdout_handle = GetStdioHandle(launch_info, STDOUT_FILENO);
    HANDLE stderr_handle = GetStdioHandle(launch_info, STDERR_FILENO);

    startupinfo.cb = sizeof(startupinfo);
    startupinfo.dwFlags |= STARTF_USESTDHANDLES;
    startupinfo.hStdError = stderr_handle;
    startupinfo.hStdInput = stdin_handle;
    startupinfo.hStdOutput = stdout_handle;

    executable = launch_info.GetExecutableFile().GetPath();
    launch_info.GetArguments().GetQuotedCommandString(commandLine);
    BOOL result = ::CreateProcessA(executable.c_str(), const_cast<char *>(commandLine.c_str()), NULL, NULL, TRUE,
                                   CREATE_NEW_CONSOLE, NULL, launch_info.GetWorkingDirectory(), &startupinfo, &pi);
    if (result)
    {
        ::CloseHandle(pi.hProcess);
        ::CloseHandle(pi.hThread);
    }

    if (stdin_handle)
        ::CloseHandle(stdin_handle);
    if (stdout_handle)
        ::CloseHandle(stdout_handle);
    if (stderr_handle)
        ::CloseHandle(stderr_handle);

    Error error;
    if (!result)
        error.SetErrorToErrno();
    return error;
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

