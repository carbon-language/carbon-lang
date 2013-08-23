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

// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Core/Error.h"
#include "lldb/Core/Log.h"
#include "lldb/Target/Process.h"

#include "lldb/Host/Host.h"
#include "lldb/Core/DataBufferHeap.h"
#include "lldb/Core/DataExtractor.h"

using namespace lldb;
using namespace lldb_private;

bool
Host::GetOSVersion(uint32_t &major,
                   uint32_t &minor,
                   uint32_t &update)
{
    OSVERSIONINFOEX info;

    ZeroMemory(&info, sizeof(OSVERSIONINFOEX));
    info.dwOSVersionInfoSize = sizeof(OSVERSIONINFOEX);

    if (GetVersionEx((LPOSVERSIONINFO) &info) == 0) {
        return false;
    }

    major = (uint32_t) info.dwMajorVersion;
    minor = (uint32_t) info.dwMinorVersion;
    update = (uint32_t) info.wServicePackMajor;

    return true;
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

uint32_t
Host::GetNumberCPUS()
{
    static uint32_t g_num_cores = UINT32_MAX;
    if (g_num_cores == UINT32_MAX)
    {
        SYSTEM_INFO system_info;
        ::GetSystemInfo(&system_info);
        g_num_cores = system_info.dwNumberOfProcessors;
    }
    return g_num_cores;
}

size_t
Host::GetPageSize()
{
    static long g_pagesize = 0;
    if (!g_pagesize)
    {
        SYSTEM_INFO systemInfo;
        GetNativeSystemInfo(&systemInfo);
        g_pagesize = systemInfo.dwPageSize;
    }
    return g_pagesize;
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
    return module_filespec;
}

void *
Host::DynamicLibraryOpen(const FileSpec &file_spec, uint32_t options, Error &error)
{
    error.SetErrorString("not implemented");
    return NULL;
}

Error
Host::DynamicLibraryClose (void *opaque)
{
    Error error;
    error.SetErrorString("not implemented");
    return error;
}

void *
Host::DynamicLibraryGetSymbol(void *opaque, const char *symbol_name, Error &error)
{
    error.SetErrorString("not implemented");
    return NULL;
}

const char *
Host::GetUserName (uint32_t uid, std::string &user_name)
{
    return NULL;
}

const char *
Host::GetGroupName (uint32_t gid, std::string &group_name)
{
    return NULL;
}

uint32_t
Host::GetUserID ()
{
    return 0;
}

uint32_t
Host::GetGroupID ()
{
    return 0;
}

uint32_t
Host::GetEffectiveUserID ()
{
    return 0;
}

uint32_t
Host::GetEffectiveGroupID ()
{
    return 0;
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