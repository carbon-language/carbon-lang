//===-- QueueItem.cpp -------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Target/Queue.h"
#include "lldb/Target/QueueItem.h"

using namespace lldb;
using namespace lldb_private;

QueueItem::QueueItem (QueueSP queue_sp) :
    m_queue_wp (),
    m_kind (eQueueItemKindUnknown),
    m_address ()
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
    return ThreadSP();
}
