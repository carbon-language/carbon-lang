//===-- HostNativeThreadBase.cpp --------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Core/Log.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Host/HostNativeThreadBase.h"
#include "lldb/Host/ThisThread.h"
#include "lldb/Host/ThreadLauncher.h"
#include "llvm/ADT/StringExtras.h"

using namespace lldb;
using namespace lldb_private;

HostNativeThreadBase::HostNativeThreadBase()
    : m_thread(LLDB_INVALID_HOST_THREAD)
    , m_state(eThreadStateInvalid)
    , m_result(0)
{
}

HostNativeThreadBase::HostNativeThreadBase(thread_t thread)
    : m_thread(thread)
    , m_state((thread == LLDB_INVALID_HOST_THREAD) ? eThreadStateInvalid : eThreadStateRunning)
    , m_result(0)
{
}

void
HostNativeThreadBase::SetState(ThreadState state)
{
    m_state = state;
}

ThreadState
HostNativeThreadBase::GetState() const
{
    return m_state;
}

lldb::thread_t
HostNativeThreadBase::GetSystemHandle() const
{
    return m_thread;
}

lldb::thread_result_t
HostNativeThreadBase::GetResult() const
{
    return m_result;
}

void
HostNativeThreadBase::Reset()
{
    m_thread = LLDB_INVALID_HOST_THREAD;
    m_state = eThreadStateInvalid;
    m_result = 0;
}

lldb::thread_t
HostNativeThreadBase::Release()
{
    lldb::thread_t result = m_thread;
    m_thread = LLDB_INVALID_HOST_THREAD;
    m_state = eThreadStateInvalid;
    m_result = 0;

    return result;
}

lldb::thread_result_t
HostNativeThreadBase::ThreadCreateTrampoline(lldb::thread_arg_t arg)
{
    ThreadLauncher::HostThreadCreateInfo *info = (ThreadLauncher::HostThreadCreateInfo *)arg;
    ThisThread::SetName(info->thread_name.c_str(), HostInfo::GetMaxThreadNameLength());

    thread_func_t thread_fptr = info->thread_fptr;
    thread_arg_t thread_arg = info->thread_arg;

    Log *log(lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_THREAD));
    if (log)
        log->Printf("thread created");

    delete info;
    return thread_fptr(thread_arg);
}
