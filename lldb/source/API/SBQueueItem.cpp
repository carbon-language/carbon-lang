//===-- SBQueueItem.cpp -----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/lldb-python.h"
#include "lldb/lldb-forward.h"

#include "lldb/API/SBAddress.h"
#include "lldb/API/SBQueueItem.h"
#include "lldb/API/SBThread.h"
#include "lldb/Core/Address.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/QueueItem.h"
#include "lldb/Target/Thread.h"

using namespace lldb;
using namespace lldb_private;

//----------------------------------------------------------------------
// Constructors
//----------------------------------------------------------------------
SBQueueItem::SBQueueItem () :
    m_queue_item_sp()
{
}

SBQueueItem::SBQueueItem (const QueueItemSP& queue_item_sp) :
    m_queue_item_sp (queue_item_sp)
{
}

//----------------------------------------------------------------------
// Destructor
//----------------------------------------------------------------------
SBQueueItem::~SBQueueItem()
{
    m_queue_item_sp.reset();
}

bool
SBQueueItem::IsValid() const
{
    return m_queue_item_sp.get() != NULL;
}


void
SBQueueItem::Clear ()
{
    m_queue_item_sp.reset();
}


void
SBQueueItem::SetQueueItem (const QueueItemSP& queue_item_sp)
{
    m_queue_item_sp = queue_item_sp;
}


lldb::QueueItemKind
SBQueueItem::GetKind () const
{
    QueueItemKind result = eQueueItemKindUnknown;
    if (m_queue_item_sp)
    {
        result = m_queue_item_sp->GetKind ();
    }
    return result;
}

void
SBQueueItem::SetKind (lldb::QueueItemKind kind)
{
    if (m_queue_item_sp)
    {
        m_queue_item_sp->SetKind (kind);
    }
}

SBAddress
SBQueueItem::GetAddress () const
{
    SBAddress result;
    if (m_queue_item_sp)
    {
        result.SetAddress (&m_queue_item_sp->GetAddress());
    }
    return result;
}

void
SBQueueItem::SetAddress (SBAddress addr)
{
    if (m_queue_item_sp)
    {
        m_queue_item_sp->SetAddress (addr.ref());
    }
}

SBThread
SBQueueItem::GetExtendedBacktraceThread (const char *type)
{
    SBThread result;
    if (m_queue_item_sp)
    {
        ProcessSP process_sp = m_queue_item_sp->GetProcessSP();
        Process::StopLocker stop_locker;
        if (process_sp && stop_locker.TryLock(&process_sp->GetRunLock()))
        {
            ThreadSP thread_sp;
            ConstString type_const (type);
            thread_sp = m_queue_item_sp->GetExtendedBacktraceThread (type_const);
            if (thread_sp)
            {
                // Save this in the Process' ExtendedThreadList so a strong pointer retains the
                // object
                process_sp->GetExtendedThreadList().AddThread (thread_sp);
                result.SetThread (thread_sp);
            }
        }
    }
    return result;
}
