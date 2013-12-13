//===-- Queue.h ------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_Queue_h_
#define liblldb_Queue_h_

#include <vector>
#include <string>

#include "lldb/lldb-forward.h"
#include "lldb/lldb-enumerations.h"
#include "lldb/lldb-private.h"


namespace lldb_private {

//------------------------------------------------------------------
// Queue:
// This class represents a libdispatch aka Grand Central Dispatch
// queue in the process.  
//
// A program using libdispatch will create queues, put work items
// (functions, blocks) on the queues.  The system will create /
// reassign pthreads to execute the work items for the queues.  A
// serial queue will be associated with a single thread (or possibly
// no thread, if it is not doing any work).  A concurrent queue may
// be associated with multiple threads.
//------------------------------------------------------------------


class Queue :
    public std::enable_shared_from_this<Queue>
{
public:

    Queue (lldb::ProcessSP process_sp, lldb::queue_id_t queue_id, const char *queue_name);

    ~Queue ();

    //------------------------------------------------------------------
    /// Get the QueueID for this Queue
    ///
    /// A 64-bit ID number that uniquely identifies a queue at this particular
    /// stop_id.  It is not guaranteed that the same QueueID will not be reused
    /// for a different queue later in the process execution after this queue
    /// has been deleted.
    ///
    /// @return
    ///     The QueueID for this Queue.
    //------------------------------------------------------------------
    lldb::queue_id_t
    GetID ();

    //------------------------------------------------------------------
    /// Get the name of this Queue
    ///
    /// @return
    ///     The name of the queue, if one is available.  
    ///     A NULL pointer is returned if none is available.
    //------------------------------------------------------------------
    const char *
    GetName ();

    //------------------------------------------------------------------
    /// Get the IndexID for this Queue
    ///
    /// An IndexID is a small integer value (starting with 1) assigned to
    /// each queue that is seen during a Process lifetime.  Ideally an
    /// IndexID will not be reused (although note that QueueID, which it is
    /// based on, is not guaranteed to be unique across the run of a program
    /// and IndexID depends on QueueID) - so if a Queue appears as IndexID 5,
    /// it will continue to show up as IndexID 5 at every process stop while
    /// that queue still exists.
    ///
    /// @return
    ///     The IndexID for this queue.
    //------------------------------------------------------------------
    uint32_t
    GetIndexID ();

    //------------------------------------------------------------------
    /// Return the threads currently associated with this queue
    ///
    /// Zero, one, or many threads may be executing code for a queue at
    /// a given point in time.  This call returns the list of threads
    /// that are currently executing work for this queue.
    ///
    /// @return
    ///     The threads currently performing work for this queue
    //------------------------------------------------------------------
    std::vector<lldb::ThreadSP>
    GetThreads ();

    //------------------------------------------------------------------
    /// Return the items that are currently enqueued
    ///
    /// "Enqueued" means that the item has been added to the queue to
    /// be done, but has not yet been done.  When the item is going to
    /// be processed it is "dequeued".
    ///
    /// @return
    ///     The vector of enqueued items for this queue
    //------------------------------------------------------------------
    const std::vector<lldb::QueueItemSP> &
    GetItems() const
    {
        return m_enqueued_items;
    }

    lldb::ProcessSP
    GetProcess() const
    {
        return m_process_wp.lock(); 
    }

protected:
    lldb::ProcessWP                 m_process_wp;
    lldb::queue_id_t                m_queue_id;
    uint32_t                        m_index_id;
    std::string                     m_queue_name;
    std::vector<lldb::QueueItemSP>  m_enqueued_items;

private:
    //------------------------------------------------------------------
    // For Queue only
    //------------------------------------------------------------------

    DISALLOW_COPY_AND_ASSIGN (Queue);

};

} // namespace lldb_private

#endif  // liblldb_Queue_h_
