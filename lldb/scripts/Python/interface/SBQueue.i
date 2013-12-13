//===-- SWIG Interface for SBQueue.h -----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

namespace lldb {

class SBQueue
{
public:
    SBQueue ();

    SBQueue (const lldb::QueueSP& queue_sp);
    
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

};

} // namespace lldb

