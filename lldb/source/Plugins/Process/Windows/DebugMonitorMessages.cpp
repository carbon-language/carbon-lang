//===-- DebugMonitorMessages.cpp --------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "DebugMonitorMessages.h"
#include "DebugMonitorMessageResults.h"

#include "lldb/Core/Error.h"
#include "lldb/Host/HostProcess.h"
#include "lldb/Target/ProcessLaunchInfo.h"

using namespace lldb;
using namespace lldb_private;

DebugMonitorMessage::DebugMonitorMessage(MonitorMessageType message_type)
    : m_message_type(message_type)
{
    Retain();
    m_completion_predicate.SetValue(nullptr, eBroadcastNever);
}

DebugMonitorMessage::~DebugMonitorMessage()
{
    const DebugMonitorMessageResult *result = m_completion_predicate.GetValue();
    if (result)
        result->Release();
    m_completion_predicate.SetValue(nullptr, eBroadcastNever);
}

const DebugMonitorMessageResult *
DebugMonitorMessage::WaitForCompletion()
{
    const DebugMonitorMessageResult *result = nullptr;
    m_completion_predicate.WaitForValueNotEqualTo(nullptr, result);
    return result;
}

void
DebugMonitorMessage::CompleteMessage(const DebugMonitorMessageResult *result)
{
    if (result)
        result->Retain();
    m_completion_predicate.SetValue(result, eBroadcastAlways);
}

LaunchProcessMessage::LaunchProcessMessage(const ProcessLaunchInfo &launch_info, lldb::ProcessSP process_plugin)
    : DebugMonitorMessage(MonitorMessageType::eLaunchProcess)
    , m_launch_info(launch_info)
    , m_process_plugin(process_plugin)
{
}

LaunchProcessMessage *
LaunchProcessMessage::Create(const ProcessLaunchInfo &launch_info, lldb::ProcessSP process_plugin)
{
    return new LaunchProcessMessage(launch_info, process_plugin);
}
