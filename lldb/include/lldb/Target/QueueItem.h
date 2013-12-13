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


class QueueItem
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

protected:
    lldb::QueueWP           m_queue_wp;
    lldb::QueueItemKind     m_kind;
    lldb_private::Address   m_address;

private:
    //------------------------------------------------------------------
    // For QueueItem only
    //------------------------------------------------------------------

    DISALLOW_COPY_AND_ASSIGN (QueueItem);

};

} // namespace lldb_private

#endif  // liblldb_QueueItem_h_
