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

#include <errno.h>
#include <pthread.h>

using namespace lldb;
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
    if (IsJoinable())
    {
        int err = ::pthread_join(m_thread, result);
        error.SetError(err, lldb::eErrorTypePOSIX);
    }
    else
    {
        if (result)
            *result = NULL;
        error.SetError(EINVAL, eErrorTypePOSIX);
    }

    Reset();
    return error;
}

Error
HostThreadPosix::Cancel()
{
    Error error;
#ifndef __ANDROID__
    int err = ::pthread_cancel(m_thread);
    error.SetError(err, eErrorTypePOSIX);
#else
    error.SetErrorString("HostThreadPosix::Cancel() not supported on Android");
#endif

    return error;
}

Error
HostThreadPosix::Detach()
{
    Error error;
    int err = ::pthread_detach(m_thread);
    error.SetError(err, eErrorTypePOSIX);
    Reset();
    return error;
}
