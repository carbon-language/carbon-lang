//===-- DebugMonitorMessageResults.cpp --------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "DebugMonitorMessageResults.h"
#include "DebugMonitorMessages.h"

#include "lldb/Core/Error.h"
#include "lldb/Host/HostProcess.h"
#include "lldb/Target/ProcessLaunchInfo.h"

using namespace lldb;
using namespace lldb_private;

DebugMonitorMessageResult::DebugMonitorMessageResult(const DebugMonitorMessage *message)
    : m_message(message)
{
    Retain();
    if (m_message)
        m_message->Retain();
}

DebugMonitorMessageResult::~DebugMonitorMessageResult()
{
    if (m_message)
        m_message->Release();
}

void
DebugMonitorMessageResult::SetError(const Error &error)
{
    m_error = error;
}

LaunchProcessMessageResult::LaunchProcessMessageResult(const LaunchProcessMessage *message)
    : DebugMonitorMessageResult(message)
{
}

LaunchProcessMessageResult *
LaunchProcessMessageResult::Create(const LaunchProcessMessage *message)
{
    return new LaunchProcessMessageResult(message);
}

void
LaunchProcessMessageResult::SetProcess(const HostProcess &process)
{
    m_process = process;
}
