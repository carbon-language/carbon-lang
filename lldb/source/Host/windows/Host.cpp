//===-- source/Host/windows/Host.cpp ----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// C Includes
#include <stdio.h>
#include "lldb/Host/windows/windows.h"
#include "lldb/Host/windows/AutoHandle.h"

// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Core/Error.h"
#include "lldb/Core/Log.h"
#include "lldb/Target/Process.h"

#include "lldb/Host/Host.h"
#include "lldb/Core/DataBufferHeap.h"
#include "lldb/Core/DataExtractor.h"
#include "lldb/Core/StreamFile.h"

// Windows includes
#include <TlHelp32.h>

using namespace lldb;
using namespace lldb_private;

namespace
{
    bool GetTripleForProcess(const FileSpec &executable, llvm::Triple &triple)
    {
        // Open the PE File as a binary file, and parse just enough information to determine the
        // machine type.
        File imageBinary(
            executable.GetPath().c_str(),
            File::eOpenOptionRead,
            lldb::eFilePermissionsUserRead);
        imageBinary.SeekFromStart(0x3c);
        int32_t peOffset = 0;
        uint32_t peHead = 0;
        uint16_t machineType = 0;
        size_t readSize = sizeof(peOffset);
        imageBinary.Read(&peOffset, readSize);
        imageBinary.SeekFromStart(peOffset);
        imageBinary.Read(&peHead, readSize);
        if (peHead != 0x00004550) // "PE\0\0", little-endian
            return false;       // Error: Can't find PE header
        readSize = 2;
        imageBinary.Read(&machineType, readSize);
        triple.setVendor(llvm::Triple::PC);
        triple.setOS(llvm::Triple::Win32);
        triple.setArch(llvm::Triple::UnknownArch);
        if (machineType == 0x8664)
            triple.setArch(llvm::Triple::x86_64);
        else if (machineType == 0x14c)
            triple.setArch(llvm::Triple::x86);

        return true;
    }

    bool GetExecutableForProcess(const AutoHandle &handle, std::string &path)
    {
        // Get the process image path.  MAX_PATH isn't long enough, paths can actually be up to 32KB.
        std::vector<char> buffer(32768);
        DWORD dwSize = buffer.size();
        if (!::QueryFullProcessImageNameA(handle.get(), 0, &buffer[0], &dwSize))
            return false;
        path.assign(&buffer[0]);
        return true;
    }

    void GetProcessExecutableAndTriple(const AutoHandle &handle, ProcessInstanceInfo &process)
    {
        // We may not have permissions to read the path from the process.  So start off by
        // setting the executable file to whatever Toolhelp32 gives us, and then try to
        // enhance this with more detailed information, but fail gracefully.
        std::string executable;
        llvm::Triple triple;
        triple.setVendor(llvm::Triple::PC);
        triple.setOS(llvm::Triple::Win32);
        triple.setArch(llvm::Triple::UnknownArch);
        if (GetExecutableForProcess(handle, executable))
        {
            FileSpec executableFile(executable.c_str(), false);
            process.SetExecutableFile(executableFile, true);
            GetTripleForProcess(executableFile, triple);
        }
        process.SetArchitecture(ArchSpec(triple));

        // TODO(zturner): Add the ability to get the process user name.
    }
}

Error
Host::LaunchProcess (ProcessLaunchInfo &launch_info)
{
    Error error;
    assert(!"Not implemented yet!!!");
    return error;
}

lldb::DataBufferSP
Host::GetAuxvData(lldb_private::Process *process)
{
    return 0;
}

std::string
Host::GetThreadName (lldb::pid_t pid, lldb::tid_t tid)
{
    return std::string();
}

lldb::tid_t
Host::GetCurrentThreadID()
{
    return lldb::tid_t(::GetCurrentThreadId());
}

lldb::thread_t
Host::GetCurrentThread ()
{
    return lldb::thread_t(::GetCurrentThread());
}

bool
Host::ThreadCancel (lldb::thread_t thread, Error *error)
{
    int err = ::TerminateThread((HANDLE)thread, 0);
    return err == 0;
}

bool
Host::ThreadDetach (lldb::thread_t thread, Error *error)
{
    return ThreadCancel(thread, error);
}

bool
Host::ThreadJoin (lldb::thread_t thread, thread_result_t *thread_result_ptr, Error *error)
{
    WaitForSingleObject((HANDLE) thread, INFINITE);
    return true;
}

lldb::thread_key_t
Host::ThreadLocalStorageCreate(ThreadLocalStorageCleanupCallback callback)
{
    return TlsAlloc();
}

void*
Host::ThreadLocalStorageGet(lldb::thread_key_t key)
{
    return ::TlsGetValue (key);
}

void
Host::ThreadLocalStorageSet(lldb::thread_key_t key, void *value)
{
   ::TlsSetValue (key, value);
}

bool
Host::SetThreadName (lldb::pid_t pid, lldb::tid_t tid, const char *name)
{
    return false;
}

bool
Host::SetShortThreadName (lldb::pid_t pid, lldb::tid_t tid,
                          const char *thread_name, size_t len)
{
    return false;
}

void
Host::Kill(lldb::pid_t pid, int signo)
{
    TerminateProcess((HANDLE) pid, 1);
}


const char *
Host::GetSignalAsCString(int signo)
{
    return NULL;
}

FileSpec
Host::GetModuleFileSpecForHostAddress (const void *host_addr)
{
    FileSpec module_filespec;

    HMODULE hmodule = NULL;
    if (!::GetModuleHandleEx(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS, (LPCTSTR)host_addr, &hmodule))
        return module_filespec;

    std::vector<char> buffer(MAX_PATH);
    DWORD chars_copied = 0;
    do {
        chars_copied = ::GetModuleFileName(hmodule, &buffer[0], buffer.size());
        if (chars_copied == buffer.size() && ::GetLastError() == ERROR_INSUFFICIENT_BUFFER)
            buffer.resize(buffer.size() * 2);
    } while (chars_copied >= buffer.size());

    module_filespec.SetFile(&buffer[0], false);
    return module_filespec;
}

uint32_t
Host::FindProcesses (const ProcessInstanceInfoMatch &match_info, ProcessInstanceInfoList &process_infos)
{
    process_infos.Clear();

    AutoHandle snapshot(CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0));
    if (!snapshot.IsValid())
        return 0;

    PROCESSENTRY32 pe = {0};
    pe.dwSize = sizeof(PROCESSENTRY32);
    if (Process32First(snapshot.get(), &pe))
    {
        do
        {
            AutoHandle handle(::OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, FALSE, pe.th32ProcessID), nullptr);

            ProcessInstanceInfo process;
            process.SetExecutableFile(FileSpec(pe.szExeFile, false), true);
            process.SetProcessID(pe.th32ProcessID);
            process.SetParentProcessID(pe.th32ParentProcessID);
            GetProcessExecutableAndTriple(handle, process);

            if (match_info.MatchAllProcesses() || match_info.Matches(process))
                process_infos.Append(process);
        } while (Process32Next(snapshot.get(), &pe));
    }
    return process_infos.GetSize();
}

bool
Host::GetProcessInfo (lldb::pid_t pid, ProcessInstanceInfo &process_info)
{
    process_info.Clear();

    AutoHandle handle(::OpenProcess(PROCESS_QUERY_INFORMATION | PROCESS_VM_READ, FALSE, pid),
                      nullptr);
    if (!handle.IsValid())
        return false;
    
    process_info.SetProcessID(pid);
    GetProcessExecutableAndTriple(handle, process_info);

    // Need to read the PEB to get parent process and command line arguments.
    return true;
}

lldb::thread_t
Host::StartMonitoringChildProcess
(
    Host::MonitorChildProcessCallback callback,
    void *callback_baton,
    lldb::pid_t pid,
    bool monitor_signals
)
{
    return LLDB_INVALID_HOST_THREAD;
}