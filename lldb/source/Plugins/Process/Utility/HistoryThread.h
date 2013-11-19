//===-- HistoryThread.h -----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_HistoryThread_h_
#define liblldb_HistoryThread_h_

#include "lldb/lldb-private.h"
#include "lldb/Host/Mutex.h"
#include "lldb/Core/Broadcaster.h"
#include "lldb/Core/Event.h"
#include "lldb/Core/UserID.h"
#include "lldb/Core/UserSettingsController.h"
#include "lldb/Target/ExecutionContextScope.h"
#include "lldb/Target/StackFrameList.h"
#include "lldb/Target/Thread.h"

namespace lldb_private {

class HistoryThread : public lldb_private::Thread
{
public:
    HistoryThread (lldb_private::Process &process, lldb::tid_t tid, std::vector<lldb::addr_t> pcs, uint32_t stop_id, bool stop_id_is_valid);

    virtual ~HistoryThread ();

    virtual lldb::RegisterContextSP
    GetRegisterContext ();

    virtual lldb::RegisterContextSP
    CreateRegisterContextForFrame (StackFrame *frame);

    virtual void
    RefreshStateAfterStop() { }

    bool
    CalculateStopInfo () { return false; }

    void 
    SetExtendedBacktraceToken (uint64_t token)
    {
        m_extended_unwind_token = token;
    }

    uint64_t
    GetExtendedBacktraceToken ()
    {
        return m_extended_unwind_token;
    }

    const char *
    GetQueueName ()
    {
        return m_queue_name.c_str();
    }

    void
    SetQueueName (const char *name)
    {
        m_queue_name = name;
    }

    lldb::queue_id_t
    GetQueueID ()
    {
        return m_queue_id;
    }

    void
    SetQueueID (lldb::queue_id_t queue)
    {
        m_queue_id = queue;
    }

    const char *
    GetThreadName ()
    {
        return m_thread_name.c_str();
    }

    uint32_t
    GetExtendedBacktraceOriginatingIndexID ();

    void
    SetThreadName (const char *name)
    {
        m_thread_name = name;
    }

protected:
    virtual lldb::StackFrameListSP
    GetStackFrameList ();

    mutable Mutex               m_framelist_mutex;
    lldb::StackFrameListSP      m_framelist;
    std::vector<lldb::addr_t>   m_pcs;
    uint32_t                    m_stop_id;
    bool                        m_stop_id_is_valid;

    uint64_t                    m_extended_unwind_token;
    std::string                 m_queue_name;
    std::string                 m_thread_name;
    lldb::tid_t                 m_originating_unique_thread_id;
    lldb::queue_id_t            m_queue_id;
};

} // namespace lldb_private

#endif  // liblldb_HistoryThread_h_
