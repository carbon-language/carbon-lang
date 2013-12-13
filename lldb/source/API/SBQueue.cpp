//===-- SBQueue.cpp ---------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/lldb-python.h"

#include "lldb/API/SBQueue.h"

#include "lldb/API/SBProcess.h"
#include "lldb/API/SBThread.h"
#include "lldb/Core/Log.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/Queue.h"
#include "lldb/Target/QueueItem.h"
#include "lldb/Target/Thread.h"

using namespace lldb;
using namespace lldb_private;

//----------------------------------------------------------------------
// Constructors
//----------------------------------------------------------------------
SBQueue::SBQueue () :
    m_queue_wp(),
    m_threads(),
    m_thread_list_fetched(false),
    m_items(),
    m_queue_items_fetched(false)
{
}

SBQueue::SBQueue (const QueueSP& queue_sp) :
    m_queue_wp(queue_sp),
    m_threads(),
    m_thread_list_fetched(false),
    m_items(),
    m_queue_items_fetched(false)
{
}

//----------------------------------------------------------------------
// Destructor
//----------------------------------------------------------------------
SBQueue::~SBQueue()
{
    m_threads.clear();
    m_items.clear();
}

bool
SBQueue::IsValid() const
{
    QueueSP queue_sp = m_queue_wp.lock();
    return queue_sp.get() != NULL;
}


void
SBQueue::Clear ()
{
    m_queue_wp.reset();
    m_thread_list_fetched = false;
    m_threads.clear();
    m_queue_items_fetched = false;
    m_items.clear();
}


void
SBQueue::SetQueue (const QueueSP& queue_sp)
{
    m_queue_wp = queue_sp;
    m_thread_list_fetched = false;
    m_threads.clear();
    m_queue_items_fetched = false;
    m_items.clear();
}

lldb::queue_id_t
SBQueue::GetQueueID () const
{
    queue_id_t result = LLDB_INVALID_QUEUE_ID;
    QueueSP queue_sp = m_queue_wp.lock();
    if (queue_sp)
    {
        result = queue_sp->GetID();
    }
    Log *log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    if (log)
        log->Printf ("SBQueue(%p)::GetQueueID () => 0x%" PRIx64, this, result);
    return result;
}

uint32_t
SBQueue::GetIndexID () const
{
    uint32_t result = LLDB_INVALID_INDEX32;
    QueueSP queue_sp = m_queue_wp.lock();
    if (queue_sp)
    {
        result = queue_sp->GetIndexID();
    }
    Log *log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    if (log)
        log->Printf ("SBQueue(%p)::GetIndexID () => %d", this, result);
    return result;
}

const char *
SBQueue::GetName () const
{
    const char *name = NULL;
    QueueSP queue_sp = m_queue_wp.lock ();
    if (queue_sp.get())
    {
        name = queue_sp->GetName();
    }

    Log *log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    if (log)
        log->Printf ("SBQueue(%p)::GetName () => %s", this, name ? name : "NULL");

    return name;
}

void
SBQueue::FetchThreads ()
{
    if (m_thread_list_fetched == false)
    {
        QueueSP queue_sp = m_queue_wp.lock();
        if (queue_sp)
        {
            Process::StopLocker stop_locker;
            if (stop_locker.TryLock (&queue_sp->GetProcess()->GetRunLock()))
            {
                const std::vector<ThreadSP> thread_list(queue_sp->GetThreads());
                m_thread_list_fetched = true;
                const uint32_t num_threads = thread_list.size();
                for (uint32_t idx = 0; idx < num_threads; ++idx)
                {
                    ThreadSP thread_sp = thread_list[idx];
                    if (thread_sp && thread_sp->IsValid())
                    {
                        m_threads.push_back (thread_sp);
                    }
                }
            }
        }
    }
}

void
SBQueue::FetchItems ()
{
    if (m_queue_items_fetched == false)
    {
        QueueSP queue_sp = m_queue_wp.lock();
        if (queue_sp)
        {
            Process::StopLocker stop_locker;
            if (stop_locker.TryLock (&queue_sp->GetProcess()->GetRunLock()))
            {
                const std::vector<QueueItemSP> queue_items(queue_sp->GetItems());
                m_queue_items_fetched = true;
                const uint32_t num_items = queue_items.size();
                for (uint32_t idx = 0; idx < num_items; ++idx)
                {
                    QueueItemSP item = queue_items[idx];
                    if (item && item->IsValid())
                    {
                        m_items.push_back (item);
                    }
                }
            }
        }
    }
}

uint32_t
SBQueue::GetNumThreads ()
{
    uint32_t result = 0;

    FetchThreads();
    if (m_thread_list_fetched)
    {
        result = m_threads.size();
    }
    return result;
}

SBThread
SBQueue::GetThreadAtIndex (uint32_t idx)
{
    FetchThreads();

    SBThread sb_thread;
    QueueSP queue_sp = m_queue_wp.lock();
    if (queue_sp && idx < m_threads.size())
    {
        ProcessSP process_sp = queue_sp->GetProcess();
        if (process_sp)
        {
            ThreadSP thread_sp = m_threads[idx].lock();
            if (thread_sp)
            {
                sb_thread.SetThread (thread_sp);
            }
        }
    }
    return sb_thread;
}


uint32_t
SBQueue::GetNumItems ()
{
    uint32_t result = 0;
    FetchItems();

    if (m_queue_items_fetched)
    {
        result = m_items.size();
    }
    return result;
}

SBQueueItem
SBQueue::GetItemAtIndex (uint32_t idx)
{
    SBQueueItem result;
    FetchItems();
    if (m_queue_items_fetched && idx < m_items.size())
    {
        result.SetQueueItem (m_items[idx]);
    }
    return result;
}

SBProcess
SBQueue::GetProcess ()
{
    SBProcess result;
    QueueSP queue_sp = m_queue_wp.lock();
    if (queue_sp)
    {
        result.SetSP (queue_sp->GetProcess());
    }
    return result;
}
