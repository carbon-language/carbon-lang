//===-- HostThreadPosix.cpp -------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Core/Error.h"
#include "lldb/Host/posix/HostThreadPosix.h"

#include <pthread.h>

using namespace lldb_private;

HostThreadPosix::HostThreadPosix()
{
}

HostThreadPosix::HostThreadPosix(lldb::thread_t thread)
    : HostNativeThreadBase(thread)
{
}

HostThreadPosix::~HostThreadPosix()
{
}

Error
HostThreadPosix::Join(lldb::thread_result_t *result)
{
    Error error;
    lldb::thread_result_t thread_result;
    int err = ::pthread_join(m_thread, &thread_result);
    error.SetError(err, lldb::eErrorTypePOSIX);
    if (err == 0)
    {
        m_state = (m_state == eThreadStateCancelling) ? eThreadStateCancelled : eThreadStateExited;
    }
    return error;
}

Error
HostThreadPosix::Cancel()
{
    Error error;
    int err = ::pthread_cancel(m_thread);
    error.SetError(err, lldb::eErrorTypePOSIX);
    if (err == 0)
        m_state = eThreadStateCancelling;

    return error;
}

Error
HostThreadPosix::Detach()
{
    Error error;
    int err = ::pthread_detach(m_thread);
    error.SetError(err, lldb::eErrorTypePOSIX);
    return error;
}
