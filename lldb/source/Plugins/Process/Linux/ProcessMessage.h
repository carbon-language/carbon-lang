//===-- ProcessMessage.h ----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_ProcessMessage_H_
#define liblldb_ProcessMessage_H_

#include <cassert>

#include "lldb/lldb-defines.h"
#include "lldb/lldb-types.h"

class ProcessMessage
{
public:

    /// The type of signal this message can correspond to.
    enum Kind
    {
        eInvalidMessage,
        eExitMessage,
        eLimboMessage,
        eSignalMessage,
        eTraceMessage,
        eBreakpointMessage
    };

    ProcessMessage()
        : m_kind(eInvalidMessage),
          m_tid(LLDB_INVALID_PROCESS_ID),
          m_data(0) { }

    Kind GetKind() const { return m_kind; }

    lldb::tid_t GetTID() const { return m_tid; }

    static ProcessMessage Exit(lldb::tid_t tid, int status) {
        return ProcessMessage(tid, eExitMessage, status);
    }

    static ProcessMessage Limbo(lldb::tid_t tid, int status) {
        return ProcessMessage(tid, eLimboMessage, status);
    }

    static ProcessMessage Signal(lldb::tid_t tid, int signum) {
        return ProcessMessage(tid, eSignalMessage, signum);
    }

    static ProcessMessage Trace(lldb::tid_t tid) {
        return ProcessMessage(tid, eTraceMessage);
    }

    static ProcessMessage Break(lldb::tid_t tid) {
        return ProcessMessage(tid, eBreakpointMessage);
    }

    int GetExitStatus() const {
        assert(GetKind() == eExitMessage || GetKind() == eLimboMessage);
        return m_data;
    }

    int GetSignal() const {
        assert(GetKind() == eSignalMessage);
        return m_data;
    }

    int GetStopStatus() const {
        assert(GetKind() == eSignalMessage);
        return m_data;
    }

private:
    ProcessMessage(lldb::tid_t tid, Kind kind, int data = 0)
        : m_kind(kind),
          m_tid(tid),
          m_data(data) { }

    Kind m_kind;
    lldb::tid_t m_tid;
    int m_data;
};

#endif // #ifndef liblldb_ProcessMessage_H_
