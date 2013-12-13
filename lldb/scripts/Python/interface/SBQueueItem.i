//===-- SWIG Interface for SBQueueItem.h ------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

namespace lldb {

class SBQueueItem
{
public:
    SBQueueItem ();

    SBQueueItem (const lldb::QueueItemSP& queue_item_sp);
    
   ~SBQueueItem();

    bool
    IsValid() const;

    void
    Clear ();

    lldb::QueueItemKind
    GetKind () const;

    void
    SetKind (lldb::QueueItemKind kind);

    lldb::SBAddress
    GetAddress () const;

    void
    SetAddress (lldb::SBAddress addr);

    void
    SetQueueItem (const lldb::QueueItemSP& queue_item_sp);

    lldb::SBThread
    GetExtendedBacktraceThread (const char *type);
};

} // namespace lldb
