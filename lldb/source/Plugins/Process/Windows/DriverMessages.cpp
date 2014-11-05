//===-- DriverMessages.cpp --------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "DriverMessages.h"
#include "DriverMessageResults.h"

#include "lldb/Core/Error.h"
#include "lldb/Host/HostProcess.h"
#include "lldb/Target/ProcessLaunchInfo.h"

using namespace lldb;
using namespace lldb_private;

DriverMessage::DriverMessage(DriverMessageType message_type)
    : m_message_type(message_type)
{
    Retain();
    m_completion_predicate.SetValue(nullptr, eBroadcastNever);
}

DriverMessage::~DriverMessage()
{
    const DriverMessageResult *result = m_completion_predicate.GetValue();
    if (result)
        result->Release();
    m_completion_predicate.SetValue(nullptr, eBroadcastNever);
}

const DriverMessageResult *
DriverMessage::WaitForCompletion()
{
    const DriverMessageResult *result = nullptr;
    m_completion_predicate.WaitForValueNotEqualTo(nullptr, result);
    return result;
}

void
DriverMessage::CompleteMessage(const DriverMessageResult *result)
{
    if (result)
        result->Retain();
    m_completion_predicate.SetValue(result, eBroadcastAlways);
}

DriverLaunchProcessMessage::DriverLaunchProcessMessage(const ProcessLaunchInfo &launch_info, DebugDelegateSP debug_delegate)
    : DriverMessage(DriverMessageType::eLaunchProcess)
    , m_launch_info(launch_info)
    , m_debug_delegate(debug_delegate)
{
}

DriverLaunchProcessMessage *
DriverLaunchProcessMessage::Create(const ProcessLaunchInfo &launch_info, DebugDelegateSP debug_delegate)
{
    return new DriverLaunchProcessMessage(launch_info, debug_delegate);
}
