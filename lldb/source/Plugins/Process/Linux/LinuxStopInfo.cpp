//===-- LinuxStopInfo.cpp ---------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "LinuxStopInfo.h"

using namespace lldb;
using namespace lldb_private;


//===----------------------------------------------------------------------===//
// LinuxLimboStopInfo

LinuxLimboStopInfo::~LinuxLimboStopInfo() { }

lldb::StopReason
LinuxLimboStopInfo::GetStopReason() const
{
    return lldb::eStopReasonTrace;
}

const char *
LinuxLimboStopInfo::GetDescription()
{
    return "thread exiting";
}

bool
LinuxLimboStopInfo::ShouldStop(Event *event_ptr)
{
    return true;
}

bool
LinuxLimboStopInfo::ShouldNotify(Event *event_ptr)
{
    return true;
}

//===----------------------------------------------------------------------===//
// LinuxCrashStopInfo

LinuxCrashStopInfo::~LinuxCrashStopInfo() { }

lldb::StopReason
LinuxCrashStopInfo::GetStopReason() const
{
    return lldb::eStopReasonException;
}

const char *
LinuxCrashStopInfo::GetDescription()
{
    return ProcessMessage::GetCrashReasonString(m_crash_reason);
}
