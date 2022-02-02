//===-- ProcessMessage.cpp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ProcessMessage.h"

using namespace lldb_private;

const char *ProcessMessage::PrintCrashReason() const {
  return CrashReasonAsString(m_crash_reason);
}

const char *ProcessMessage::PrintKind(Kind kind) {
  const char *str = nullptr;

  switch (kind) {
  case eInvalidMessage:
    str = "eInvalidMessage";
    break;
  case eAttachMessage:
    str = "eAttachMessage";
    break;
  case eExitMessage:
    str = "eExitMessage";
    break;
  case eLimboMessage:
    str = "eLimboMessage";
    break;
  case eSignalMessage:
    str = "eSignalMessage";
    break;
  case eSignalDeliveredMessage:
    str = "eSignalDeliveredMessage";
    break;
  case eTraceMessage:
    str = "eTraceMessage";
    break;
  case eBreakpointMessage:
    str = "eBreakpointMessage";
    break;
  case eWatchpointMessage:
    str = "eWatchpointMessage";
    break;
  case eCrashMessage:
    str = "eCrashMessage";
    break;
  case eNewThreadMessage:
    str = "eNewThreadMessage";
    break;
  case eExecMessage:
    str = "eExecMessage";
    break;
  }
  return str;
}

const char *ProcessMessage::PrintKind() const { return PrintKind(m_kind); }
