//===-- SBQueue.h -----------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SBQueue_h_
#define LLDB_SBQueue_h_

#include <vector>

#include "lldb/lldb-forward.h"
#include "lldb/API/SBDefines.h"
#include "lldb/API/SBQueueItem.h"

namespace lldb {

class SBQueue
{
public:
    SBQueue ();

    SBQueue (const QueueSP& queue_sp);
    
   ~SBQueue();

    bool
    IsValid() const;

    void
    Clear ();

    lldb::SBProcess
    GetProcess ();

    lldb::queue_id_t
    GetQueueID () const;

    const char *
    GetName () const;

    uint32_t
    GetIndexID () const;

    uint32_t
    GetNumThreads ();

    lldb::SBThread
    GetThreadAtIndex (uint32_t);

    uint32_t
    GetNumItems ();

    lldb::SBQueueItem
    GetItemAtIndex (uint32_t);

protected:
    friend class SBProcess;

    void
    SetQueue (const lldb::QueueSP& queue_sp);

    void
    FetchThreads ();

    void
    FetchItems ();

private:
    lldb::QueueWP                   m_queue_wp;
    std::vector<lldb::ThreadWP>     m_threads;              // threads currently executing this queue's items
    bool                            m_thread_list_fetched;  // have we tried to fetch the threads list already?
    std::vector<lldb::QueueItemSP>  m_items;       // items currently enqueued
    bool                            m_queue_items_fetched;  // have we tried to fetch the item list already?
};

} // namespace lldb

#endif  // LLDB_SBQueue_h_
