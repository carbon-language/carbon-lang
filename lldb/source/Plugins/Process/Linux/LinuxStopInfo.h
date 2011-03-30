//===-- LinuxStopInfo.h -----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_LinuxStopInfo_H_
#define liblldb_LinuxStopInfo_H_

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Target/StopInfo.h"

#include "LinuxThread.h"
#include "ProcessMessage.h"

//===----------------------------------------------------------------------===//
/// @class LinuxStopInfo
/// @brief Simple base class for all Linux-specific StopInfo objects.
///
class LinuxStopInfo
    : public lldb_private::StopInfo
{
public:
    LinuxStopInfo(lldb_private::Thread &thread, uint32_t status)
        : StopInfo(thread, status)
        { }
};

//===----------------------------------------------------------------------===//
/// @class LinuxLimboStopInfo
/// @brief Represents the stop state of a process ready to exit.
///
class LinuxLimboStopInfo
    : public LinuxStopInfo
{
public:
    LinuxLimboStopInfo(LinuxThread &thread)
        : LinuxStopInfo(thread, 0)
        { }

    ~LinuxLimboStopInfo();

    lldb::StopReason
    GetStopReason() const;

    const char *
    GetDescription();

    bool
    ShouldStop(lldb_private::Event *event_ptr);

    bool
    ShouldNotify(lldb_private::Event *event_ptr);
};


//===----------------------------------------------------------------------===//
/// @class LinuxCrashStopInfo
/// @brief Represents the stop state of process that is ready to crash.
///
class LinuxCrashStopInfo
    : public LinuxStopInfo
{
public:
    LinuxCrashStopInfo(LinuxThread &thread, uint32_t status, 
                  ProcessMessage::CrashReason reason)
        : LinuxStopInfo(thread, status),
          m_crash_reason(reason)
        { }

    ~LinuxCrashStopInfo();

    lldb::StopReason
    GetStopReason() const;

    const char *
    GetDescription();

    ProcessMessage::CrashReason
    GetCrashReason() const;

private:
    ProcessMessage::CrashReason m_crash_reason;
};    

#endif
