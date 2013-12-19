//===-- Queue.cpp -----------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Target/Process.h"
#include "lldb/Target/Queue.h"
#include "lldb/Target/QueueList.h"
#include "lldb/Target/Thread.h"

using namespace lldb;
using namespace lldb_private;

Queue::Queue (ProcessSP process_sp, lldb::queue_id_t queue_id, const char *queue_name) :
    m_process_wp (),
    m_queue_id (queue_id),
    m_queue_name (),
    m_enqueued_items()
{
    if (queue_name)
        m_queue_name = queue_name;

    m_process_wp = process_sp;
    m_index_id = process_sp->AssignIndexIDToQueue (queue_id);
}

Queue::~Queue ()
{
}

queue_id_t
Queue::GetID ()
{
    return m_queue_id;
}

const char *
Queue::GetName ()
{
    const char *result = NULL;
    if (m_queue_name.size() > 0)
        result = m_queue_name.c_str();
    return result;
}

uint32_t
Queue::GetIndexID ()
{
    return m_index_id;
}

std::vector<lldb::ThreadSP>
Queue::GetThreads ()
{
    std::vector<ThreadSP> result;
    ProcessSP process_sp = m_process_wp.lock();
    if (process_sp.get ())
    {
        for (ThreadSP thread_sp : process_sp->Threads())
        {
            if (thread_sp->GetQueueID() == m_queue_id)
            {
                result.push_back (thread_sp);
            }
        }
    }
    return result;
}
