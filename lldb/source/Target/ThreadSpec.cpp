//===-- ThreadSpec.cpp ----------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Target/Thread.h"
#include "lldb/Target/ThreadSpec.h"

using namespace lldb;
using namespace lldb_private;

ThreadSpec::ThreadSpec() :
    m_index (UINT32_MAX),
    m_tid (LLDB_INVALID_THREAD_ID),
    m_name(),
    m_queue_name ()
{
}

ThreadSpec::ThreadSpec (const ThreadSpec &rhs) :
    m_index(rhs.m_index),
    m_tid(rhs.m_tid),
    m_name(rhs.m_name),
    m_queue_name(rhs.m_queue_name)
{
}

const ThreadSpec &
ThreadSpec::operator=(const ThreadSpec &rhs)
{
    m_index = rhs.m_index;
    m_tid = rhs.m_tid;
    m_name = rhs.m_name;
    m_queue_name = rhs.m_queue_name;
    return *this;
}

const char *
ThreadSpec::GetName () const
{
    if (m_name.empty())
        return NULL;
    else
        return m_name.c_str();
}

const char *
ThreadSpec::GetQueueName () const
{
    if (m_queue_name.empty())
        return NULL;
    else
        return m_queue_name.c_str();
}

bool
ThreadSpec::ThreadPassesBasicTests (Thread *thread) const
{

    if (!HasSpecification())
        return true;
        
    if (!TIDMatches(thread->GetID()))
        return false;
        
    if (!IndexMatches(thread->GetIndexID()))
        return false;
        
    if (!NameMatches (thread->GetName()))
        return false;
        
    if (!QueueNameMatches (thread->GetQueueName()))
        return false;
        
    return true;

}

bool
ThreadSpec::HasSpecification() const
{
    return (m_index != UINT32_MAX || m_tid != LLDB_INVALID_THREAD_ID || !m_name.empty() || !m_queue_name.empty());
}
void
ThreadSpec::GetDescription (Stream *s, lldb::DescriptionLevel level) const
{
    if (!HasSpecification())
    {
        if (level == eDescriptionLevelBrief)
        {
            s->PutCString("thread spec: no ");
        }
    }
    else
    {
        if (level == eDescriptionLevelBrief)
        {
            s->PutCString("thread spec: yes ");
        }
        else
        {
            if (GetTID() != LLDB_INVALID_THREAD_ID)
                s->Printf("tid: 0x%llx ", GetTID());
                
            if (GetIndex() != UINT32_MAX)
                s->Printf("index: %d ", GetIndex());
                
            const char *name = GetName();
            if (name)
                s->Printf ("thread name: \"%s\" ", name);
                
            const char *queue_name = GetQueueName();
            if (queue_name)
                s->Printf ("queue name: \"%s\" ", queue_name);
        }

    }
}
