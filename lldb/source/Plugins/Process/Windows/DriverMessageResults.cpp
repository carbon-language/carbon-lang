//===-- DriverMessageResults.cpp --------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "DriverMessageResults.h"
#include "DriverMessages.h"

#include "lldb/Core/Error.h"
#include "lldb/Host/HostProcess.h"
#include "lldb/Target/ProcessLaunchInfo.h"

using namespace lldb;
using namespace lldb_private;

DriverMessageResult::DriverMessageResult(const DriverMessage *message)
    : m_message(message)
{
    Retain();
    if (m_message)
        m_message->Retain();
}

DriverMessageResult::~DriverMessageResult()
{
    if (m_message)
        m_message->Release();
}

void
DriverMessageResult::SetError(const Error &error)
{
    m_error = error;
}

DriverLaunchProcessMessageResult::DriverLaunchProcessMessageResult(const DriverLaunchProcessMessage *message)
    : DriverMessageResult(message)
{
}

DriverLaunchProcessMessageResult *
DriverLaunchProcessMessageResult::Create(const DriverLaunchProcessMessage *message)
{
    return new DriverLaunchProcessMessageResult(message);
}

void
DriverLaunchProcessMessageResult::SetProcess(const HostProcess &process)
{
    m_process = process;
}
