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

namespace lldb_private
{

    class QueueImpl
    {
    public:
        QueueImpl () :
            m_queue_wp(),
            m_threads(),
            m_thread_list_fetched(false),
            m_items(),
            m_queue_items_fetched(false)
        {
        }

        QueueImpl (const lldb::QueueSP &queue_sp) :
            m_queue_wp(),
            m_threads(),
            m_thread_list_fetched(false),
            m_items(),
            m_queue_items_fetched(false)
        {
            m_queue_wp = queue_sp;
        }

        QueueImpl (const QueueImpl &rhs)
        {
            if (&rhs == this)
                return;
            m_queue_wp = rhs.m_queue_wp;
            m_threads = rhs.m_threads;
            m_thread_list_fetched = rhs.m_thread_list_fetched;
            m_items = rhs.m_items;
            m_queue_items_fetched = rhs.m_queue_items_fetched;
        }

        ~QueueImpl ()
        {
        }

        bool
        IsValid ()
        {
            return m_queue_wp.lock() != NULL;
        }

        void
        Clear ()
        {
            m_queue_wp.reset();
            m_thread_list_fetched = false;
            m_threads.clear();
            m_queue_items_fetched = false;
            m_items.clear();
        }

        void
        SetQueue (const lldb::QueueSP &queue_sp)
        {
            Clear();
            m_queue_wp = queue_sp;
        }

        lldb::queue_id_t
        GetQueueID () const
        {
            lldb::queue_id_t result = LLDB_INVALID_QUEUE_ID;
            lldb::QueueSP queue_sp = m_queue_wp.lock();
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
        GetIndexID () const
        {
            uint32_t result = LLDB_INVALID_INDEX32;
            lldb::QueueSP queue_sp = m_queue_wp.lock();
            if (queue_sp)
            {
                result = queue_sp->GetIndexID();
            }
            Log *log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
            if (log)
                log->Printf ("SBQueueImpl(%p)::GetIndexID () => %d", this, result);
            return result;
        }
        
        const char *
        GetName () const
        {
            const char *name = NULL;
            lldb::QueueSP queue_sp = m_queue_wp.lock ();
            if (queue_sp.get())
            {
                name = queue_sp->GetName();
            }
        
            Log *log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
            if (log)
                log->Printf ("SBQueueImpl(%p)::GetName () => %s", this, name ? name : "NULL");
        
            return name;
        }
        
        void
        FetchThreads ()
        {
            if (m_thread_list_fetched == false)
            {
                lldb::QueueSP queue_sp = m_queue_wp.lock();
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
        FetchItems ()
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
        GetNumThreads ()
        {
            uint32_t result = 0;
        
            FetchThreads();
            if (m_thread_list_fetched)
            {
                result = m_threads.size();
            }
            return result;
        }
        
        lldb::SBThread
        GetThreadAtIndex (uint32_t idx)
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
        GetNumItems ()
        {
            uint32_t result = 0;
            FetchItems();
        
            if (m_queue_items_fetched)
            {
                result = m_items.size();
            }
            return result;
        }
        
        lldb::SBQueueItem
        GetItemAtIndex (uint32_t idx)
        {
            SBQueueItem result;
            FetchItems();
            if (m_queue_items_fetched && idx < m_items.size())
            {
                result.SetQueueItem (m_items[idx]);
            }
            return result;
        }
        
        lldb::SBProcess
        GetProcess ()
        {
            SBProcess result;
            QueueSP queue_sp = m_queue_wp.lock();
            if (queue_sp)
            {
                result.SetSP (queue_sp->GetProcess());
            }
            return result;
        }

    private:
        lldb::QueueWP                   m_queue_wp;
        std::vector<lldb::ThreadWP>     m_threads;              // threads currently executing this queue's items
        bool                            m_thread_list_fetched;  // have we tried to fetch the threads list already?
        std::vector<lldb::QueueItemSP>  m_items;       // items currently enqueued
        bool                            m_queue_items_fetched;  // have we tried to fetch the item list already?
    };

}

SBQueue::SBQueue () :
    m_opaque_sp (new QueueImpl())
{
}

SBQueue::SBQueue (const QueueSP& queue_sp) :
    m_opaque_sp (new QueueImpl (queue_sp))
{
}

SBQueue::SBQueue (const SBQueue &rhs)
{
    if (&rhs == this)
        return;

    m_opaque_sp = rhs.m_opaque_sp;
}

const lldb::SBQueue &
SBQueue::operator = (const lldb::SBQueue &rhs)
{
    m_opaque_sp = rhs.m_opaque_sp;
    return *this;
}

SBQueue::~SBQueue()
{
}

bool
SBQueue::IsValid() const
{
    return m_opaque_sp->IsValid();
}


void
SBQueue::Clear ()
{
    m_opaque_sp->Clear();
}


void
SBQueue::SetQueue (const QueueSP& queue_sp)
{
    m_opaque_sp->SetQueue (queue_sp);
}

lldb::queue_id_t
SBQueue::GetQueueID () const
{
    return m_opaque_sp->GetQueueID ();
}

uint32_t
SBQueue::GetIndexID () const
{
    return m_opaque_sp->GetIndexID ();
}

const char *
SBQueue::GetName () const
{
    return m_opaque_sp->GetName ();
}

uint32_t
SBQueue::GetNumThreads ()
{
    return m_opaque_sp->GetNumThreads ();
}

SBThread
SBQueue::GetThreadAtIndex (uint32_t idx)
{
    return m_opaque_sp->GetThreadAtIndex (idx);
}


uint32_t
SBQueue::GetNumItems ()
{
    return m_opaque_sp->GetNumItems ();
}

SBQueueItem
SBQueue::GetItemAtIndex (uint32_t idx)
{
    return m_opaque_sp->GetItemAtIndex (idx);
}

SBProcess
SBQueue::GetProcess ()
{
    return m_opaque_sp->GetProcess();
}
