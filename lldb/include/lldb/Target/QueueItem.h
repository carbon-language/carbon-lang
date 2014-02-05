//===-- QueueItem.h ---------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_QueueItem_h_
#define liblldb_QueueItem_h_

#include <vector>

#include "lldb/lldb-private.h"
#include "lldb/lldb-enumerations.h"

#include "lldb/Core/Address.h"
#include "lldb/Core/ConstString.h"


namespace lldb_private {

//------------------------------------------------------------------
// QueueItem:
// This class represents a work item enqueued on a libdispatch aka
// Grand Central Dispatch (GCD) queue.  Most often, this will be a
// function or block.
// "enqueued" here means that the work item has been added to a queue
// but it has not yet started executing.  When it is "dequeued", 
// execution of the item begins.
//------------------------------------------------------------------


class QueueItem :
    public std::enable_shared_from_this<QueueItem>
{
public:

    QueueItem (lldb::QueueSP queue_sp);

    ~QueueItem ();

    //------------------------------------------------------------------
    /// Get the kind of work item this is
    ///
    /// @return
    ///     The type of work item that this QueueItem object
    ///     represents.  eQueueItemKindUnknown may be returned.
    //------------------------------------------------------------------
    lldb::QueueItemKind
    GetKind () const;

    //------------------------------------------------------------------
    /// Set the type of work item this is
    ///
    /// @param [in] item_kind
    ///     Set the kind of this work item object.
    //------------------------------------------------------------------
    void
    SetKind (lldb::QueueItemKind item_kind);

    //------------------------------------------------------------------
    /// Get the code address that will be executed when this work item
    /// is executed.
    ///
    /// @return
    ///     The address that will be invoked when this work item is
    ///     executed.  Not all types of QueueItems will have an
    ///     address associated with them; check that the returned 
    ///     Address is valid, or check that the WorkItemKind is a
    ///     kind that involves an address, such as eQueueItemKindFunction
    ///     or eQueueItemKindBlock.
    //------------------------------------------------------------------
    lldb_private::Address &
    GetAddress ();

    //------------------------------------------------------------------
    /// Set the work item address for this object
    ///
    /// @param [in] addr
    ///     The address that will be invoked when this work item
    ///     is executed.
    //------------------------------------------------------------------
    void
    SetAddress (lldb_private::Address addr);

    //------------------------------------------------------------------
    /// Check if this QueueItem object is valid
    ///
    /// If the weak pointer to the parent Queue cannot be revivified,
    /// it is invalid.
    ///
    /// @return
    ///     True if this object is valid.
    //------------------------------------------------------------------
    bool
    IsValid ()
    {
        return m_queue_wp.lock() != NULL;
    }

    //------------------------------------------------------------------
    /// Get an extended backtrace thread for this queue item, if available
    ///
    /// If the backtrace/thread information was collected when this item
    /// was enqueued, this call will provide it.
    ///
    /// @param [in] type
    ///     The type of extended backtrace being requested, e.g. "libdispatch"
    ///     or "pthread".
    ///
    /// @return 
    ///     A thread shared pointer which will have a reference to an extended
    ///     thread if one was available.
    //------------------------------------------------------------------
    lldb::ThreadSP
    GetExtendedBacktraceThread (ConstString type);

    void
    SetItemThatEnqueuedThis (lldb::addr_t address_of_item)
    {
        m_item_that_enqueued_this_ref = address_of_item;
    }

    lldb::addr_t
    GetItemThatEnqueuedThis ()
    {
        return m_item_that_enqueued_this_ref;
    }

    void
    SetEnqueueingThreadID (lldb::tid_t tid)
    {
        m_enqueueing_thread_id = tid;
    }

    lldb::tid_t
    GetEnqueueingThreadID ()
    {
        return m_enqueueing_thread_id;
    }

    void
    SetEnqueueingQueueID (lldb::queue_id_t qid)
    {
        m_enqueueing_queue_id = qid;
    }

    lldb::queue_id_t
    GetEnqueueingQueueID ()
    {
        return m_enqueueing_queue_id;
    }

    void
    SetTargetQueueID (lldb::queue_id_t qid)
    {
        m_target_queue_id = qid;
    }

    void
    SetStopID (uint32_t stop_id)
    {
        m_stop_id = stop_id;
    }

    uint32_t
    GetStopID ()
    {
        return m_stop_id;
    }

    void
    SetEnqueueingBacktrace (std::vector<lldb::addr_t> backtrace)
    {
        m_backtrace = backtrace;
    }

    std::vector<lldb::addr_t> &
    GetEnqueueingBacktrace ()
    {
        return m_backtrace;
    }

    void
    SetThreadLabel (std::string thread_name)
    {
        m_thread_label = thread_name;
    }

    std::string
    GetThreadLabel ()
    {
        return m_thread_label;
    }

    void
    SetQueueLabel (std::string queue_name)
    {
        m_queue_label = queue_name;
    }

    std::string
    GetQueueLabel ()
    {
        return m_queue_label;
    }

    void
    SetTargetQueueLabel (std::string queue_name)
    {
        m_target_queue_label = queue_name;
    }

protected:
    lldb::QueueWP           m_queue_wp;
    lldb::QueueItemKind     m_kind;
    lldb_private::Address   m_address;

    lldb::addr_t            m_item_that_enqueued_this_ref;  // a handle that we can pass into libBacktraceRecording
                                                            // to get the QueueItem that enqueued this item
    lldb::tid_t             m_enqueueing_thread_id;    // thread that enqueued this item
    lldb::queue_id_t        m_enqueueing_queue_id;     // Queue that enqueued this item, if it was a queue
    lldb::queue_id_t        m_target_queue_id;
    uint32_t                m_stop_id;                 // indicates when this backtrace was recorded in time
    std::vector<lldb::addr_t>    m_backtrace;
    std::string             m_thread_label;
    std::string             m_queue_label;
    std::string             m_target_queue_label;


private:
    //------------------------------------------------------------------
    // For QueueItem only
    //------------------------------------------------------------------

    DISALLOW_COPY_AND_ASSIGN (QueueItem);

};

} // namespace lldb_private

#endif  // liblldb_QueueItem_h_
