//===-- StopInfo.h ----------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_StopInfo_h_
#define liblldb_StopInfo_h_

// C Includes
// C++ Includes
#include <string>

// Other libraries and framework includes
// Project includes
#include "lldb/lldb-public.h"

namespace lldb_private {

class StopInfo
{
public:
    //------------------------------------------------------------------
    // Constructors and Destructors
    //------------------------------------------------------------------
    StopInfo (Thread &thread, uint64_t value);

    virtual ~StopInfo()
    {
    }


    bool
    IsValid () const;

    Thread &
    GetThread()
    {
        return m_thread;
    }

    const Thread &
    GetThread() const
    {
        return m_thread;
    }

    // The value of the StopInfo depends on the StopReason.
    // StopReason                  Meaning
    // ----------------------------------------------
    // eStopReasonBreakpoint       BreakpointSiteID
    // eStopReasonSignal           Signal number
    // eStopReasonWatchpoint       WatchpointSiteID
    // eStopReasonPlanComplete     No significance
    
    uint64_t
    GetValue() const
    {
        return m_value;
    }

    virtual lldb::StopReason
    GetStopReason () const = 0;
    
    // Perform any action that is associated with this stop.  This is done as the
    // Event is removed from the event queue.
    virtual void
    PerformAction (Event *event_ptr)
    {
    }

    // Stop the thread by default. Subclasses can override this to allow
    // the thread to continue if desired.
    virtual bool
    ShouldStop (Event *event_ptr)
    {
        return true;
    }

    // If should stop returns false, check if we should notify of this event
    virtual bool
    ShouldNotify (Event *event_ptr)
    {
        return false;
    }

    virtual void
    WillResume (lldb::StateType resume_state)
    {
        // By default, don't do anything
    }
    
    virtual const char *
    GetDescription ()
    {
        return m_description.c_str();
    }

    virtual void
    SetDescription (const char *desc_cstr)
    {
        if (desc_cstr && desc_cstr[0])
            m_description.assign (desc_cstr);
        else
            m_description.clear();
    }

    static lldb::StopInfoSP
    CreateStopReasonWithBreakpointSiteID (Thread &thread, lldb::break_id_t break_id);

    // This creates a StopInfo for the thread where the should_stop is already set, and won't be recalculated.
    static lldb::StopInfoSP
    CreateStopReasonWithBreakpointSiteID (Thread &thread, lldb::break_id_t break_id, bool should_stop);

    static lldb::StopInfoSP
    CreateStopReasonWithWatchpointID (Thread &thread, lldb::break_id_t watch_id);

    static lldb::StopInfoSP
    CreateStopReasonWithSignal (Thread &thread, int signo);

    static lldb::StopInfoSP
    CreateStopReasonToTrace (Thread &thread);

    static lldb::StopInfoSP
    CreateStopReasonWithPlan (lldb::ThreadPlanSP &plan);

    static lldb::StopInfoSP
    CreateStopReasonWithException (Thread &thread, const char *description);

protected:
    //------------------------------------------------------------------
    // Classes that inherit from StackID can see and modify these
    //------------------------------------------------------------------
    Thread &        m_thread;   // The thread corresponding to the stop reason.
    uint32_t        m_stop_id;  // The process stop ID for which this stop info is valid
    uint64_t        m_value;    // A generic value that can be used for things pertaining to this stop info
    std::string     m_description; // A textual description describing this stop.
    
    // This provides an accessor to the PrivateEventState of the process for StopInfo's w/o having to make each
    // StopInfo subclass a friend of Process.
    lldb::StateType
    GetPrivateState ();

private:
    friend class Thread;
    
    // MakeStopInfoValid is necessary to allow saved stop infos to resurrect themselves as valid.  It should
    // only need to be called by Thread::RestoreThreadStateFromCheckpoint.
    void
    MakeStopInfoValid ();
    
    DISALLOW_COPY_AND_ASSIGN (StopInfo);
};

} // namespace lldb_private

#endif  // liblldb_StopInfo_h_
