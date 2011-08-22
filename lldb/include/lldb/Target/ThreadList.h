//===-- ThreadList.h --------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_ThreadList_h_
#define liblldb_ThreadList_h_

#include <vector>

#include "lldb/lldb-private.h"
#include "lldb/Host/Mutex.h"
#include "lldb/Core/UserID.h"


// FIXME: Currently this is a thread list with lots of functionality for use only by
// the process for which this is the thread list.  If we ever want a container class
// to hand out that is just a random subset of threads, with iterator functionality,
// then we should make that part a base class, and make a ProcessThreadList for the
// process.
namespace lldb_private {

class ThreadList
{
friend class Process;

public:

    ThreadList (Process *process);

    ThreadList (const ThreadList &rhs);

    ~ThreadList ();

    const ThreadList&
    operator = (const ThreadList& rhs);

    uint32_t
    GetSize(bool can_update = true);

    void
    AddThread (lldb::ThreadSP &thread_sp);

    lldb::ThreadSP
    GetSelectedThread ();

    bool
    SetSelectedThreadByID (lldb::tid_t tid);

    bool
    SetSelectedThreadByIndexID (uint32_t index_id);

    void
    Clear();

    // Note that "idx" is not the same as the "thread_index". It is a zero
    // based index to accessing the current threads, whereas "thread_index"
    // is a unique index assigned
    lldb::ThreadSP
    GetThreadAtIndex (uint32_t idx, bool can_update = true);

    lldb::ThreadSP
    FindThreadByID (lldb::tid_t tid, bool can_update = true);

    lldb::ThreadSP
    FindThreadByIndexID (lldb::tid_t index_id, bool can_update = true);

    lldb::ThreadSP
    GetThreadSPForThreadPtr (Thread *thread_ptr);

    bool
    ShouldStop (Event *event_ptr);

    Vote
    ShouldReportStop (Event *event_ptr);

    Vote
    ShouldReportRun (Event *event_ptr);

    void
    RefreshStateAfterStop ();

    bool
    WillResume ();

    void
    DidResume ();

    void
    DiscardThreadPlans();

    uint32_t
    GetStopID () const;

    void
    SetStopID (uint32_t stop_id);

    Mutex &
    GetMutex ()
    {
        return m_threads_mutex;
    }
    
    void
    Update (ThreadList &rhs);
    
protected:

    typedef std::vector<lldb::ThreadSP> collection;
    //------------------------------------------------------------------
    // Classes that inherit from Process can see and modify these
    //------------------------------------------------------------------
    Process *m_process; ///< The process that manages this thread list.
    uint32_t m_stop_id; ///< The process stop ID that this thread list is valid for.
    collection m_threads; ///< The threads for this process.
    mutable Mutex m_threads_mutex;
    lldb::tid_t m_selected_tid;  ///< For targets that need the notion of a current thread.

private:
    ThreadList ();
};

} // namespace lldb_private

#endif  // liblldb_ThreadList_h_
