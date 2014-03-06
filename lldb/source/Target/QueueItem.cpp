//===-- QueueItem.cpp -------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Target/Queue.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/QueueItem.h"
#include "lldb/Target/SystemRuntime.h"

using namespace lldb;
using namespace lldb_private;

QueueItem::QueueItem (QueueSP queue_sp) :
    m_queue_wp (),
    m_kind (eQueueItemKindUnknown),
    m_address (),
    m_item_that_enqueued_this_ref (LLDB_INVALID_ADDRESS),
    m_enqueueing_thread_id (LLDB_INVALID_THREAD_ID),
    m_enqueueing_queue_id (LLDB_INVALID_QUEUE_ID),
    m_target_queue_id (LLDB_INVALID_QUEUE_ID),
    m_stop_id (0),
    m_backtrace(),
    m_thread_label(),
    m_queue_label(),
    m_target_queue_label()
{
    m_queue_wp = queue_sp;
}

QueueItem::~QueueItem ()
{
}

QueueItemKind
QueueItem::GetKind() const
{
    return m_kind;
}

void
QueueItem::SetKind (QueueItemKind item_kind)
{
    m_kind = item_kind;
}

Address &
QueueItem::GetAddress () 
{
    return m_address;
}

void
QueueItem::SetAddress (Address addr)
{
    m_address = addr;
}

ThreadSP
QueueItem::GetExtendedBacktraceThread (ConstString type)
{
    ThreadSP return_thread;
    QueueSP queue_sp = m_queue_wp.lock();
    if (queue_sp)
    {
        ProcessSP process_sp = queue_sp->GetProcess();
        if (process_sp && process_sp->GetSystemRuntime())
        {
            return_thread = process_sp->GetSystemRuntime()->GetExtendedBacktraceForQueueItem (this->shared_from_this(), type);
        }
    }
    return return_thread;
}

ProcessSP
QueueItem::GetProcessSP()
{
    ProcessSP process_sp;
    QueueSP queue_sp = m_queue_wp.lock ();
    if (queue_sp)
    {
        process_sp = queue_sp->GetProcess();
    }
    return process_sp;
}
