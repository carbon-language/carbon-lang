//===-- HistoryThread.cpp ---------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/lldb-private.h"

#include "Plugins/Process/Utility/HistoryUnwind.h"
#include "Plugins/Process/Utility/HistoryThread.h"
#include "Plugins/Process/Utility/RegisterContextHistory.h"

#include "lldb/Target/StackFrameList.h"
#include "lldb/Target/Process.h"

using namespace lldb;
using namespace lldb_private;

HistoryThread::HistoryThread (lldb_private::Process &process, 
                              std::vector<lldb::addr_t> pcs, 
                              uint32_t stop_id, 
                              bool stop_id_is_valid) : 
        Thread (process, LLDB_INVALID_THREAD_ID),
        m_framelist_mutex(),
        m_framelist(),
        m_pcs (pcs),
        m_stop_id (stop_id),
        m_stop_id_is_valid (stop_id_is_valid)
{
    m_unwinder_ap.reset (new HistoryUnwind (*this, pcs, stop_id, stop_id_is_valid));
}

HistoryThread::~HistoryThread ()
{
}

lldb::RegisterContextSP
HistoryThread::GetRegisterContext ()
{
    RegisterContextSP rctx ;
    if (m_pcs.size() > 0)
    {
        rctx.reset (new RegisterContextHistory (*this, 0, GetProcess()->GetAddressByteSize(), m_pcs[0]));
    }
    return rctx;

}

lldb::RegisterContextSP
HistoryThread::CreateRegisterContextForFrame (StackFrame *frame)
{
    return m_unwinder_ap->CreateRegisterContextForFrame (frame);
}

lldb::StackFrameListSP
HistoryThread::GetStackFrameList ()
{
    Mutex::Locker (m_framelist_mutex);
    if (m_framelist.get() == NULL)
    {
        m_framelist.reset (new StackFrameList (*this, StackFrameListSP(), true));
    }

    return m_framelist;
}
