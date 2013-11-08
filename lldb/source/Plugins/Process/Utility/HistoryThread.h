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
    HistoryThread (lldb_private::Process &process, std::vector<lldb::addr_t> pcs, uint32_t stop_id, bool stop_id_is_valid);

    virtual ~HistoryThread ();

    virtual lldb::RegisterContextSP
    GetRegisterContext ();

    virtual lldb::RegisterContextSP
    CreateRegisterContextForFrame (StackFrame *frame);

    virtual void
    RefreshStateAfterStop() { }

    bool
    CalculateStopInfo () { return false; }

protected:
    virtual lldb::StackFrameListSP
    GetStackFrameList ();

    mutable Mutex               m_framelist_mutex;
    lldb::StackFrameListSP      m_framelist;
    std::vector<lldb::addr_t>   m_pcs;
    uint32_t                    m_stop_id;
    bool                        m_stop_id_is_valid;
};

} // namespace lldb_private

#endif  // liblldb_HistoryThread_h_
