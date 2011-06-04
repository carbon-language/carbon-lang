//===-- StopInfo.cpp ---------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Target/StopInfo.h"

// C Includes
// C++ Includes
#include <string>

// Other libraries and framework includes
// Project includes
#include "lldb/Core/Log.h"
#include "lldb/Breakpoint/Breakpoint.h"
#include "lldb/Breakpoint/BreakpointLocation.h"
#include "lldb/Breakpoint/StoppointCallbackContext.h"
#include "lldb/Core/StreamString.h"
#include "lldb/Target/Thread.h"
#include "lldb/Target/ThreadPlan.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/UnixSignals.h"

using namespace lldb;
using namespace lldb_private;

StopInfo::StopInfo (Thread &thread, uint64_t value) :
    m_thread (thread),
    m_stop_id (thread.GetProcess().GetStopID()),
    m_value (value)
{
}

bool
StopInfo::IsValid () const
{
    return m_thread.GetProcess().GetStopID() == m_stop_id;
}

void
StopInfo::MakeStopInfoValid ()
{
    m_stop_id = m_thread.GetProcess().GetStopID();
}

//----------------------------------------------------------------------
// StopInfoBreakpoint
//----------------------------------------------------------------------

class StopInfoBreakpoint : public StopInfo
{
public:

    StopInfoBreakpoint (Thread &thread, break_id_t break_id) :
        StopInfo (thread, break_id),
        m_description(),
        m_should_stop (false),
        m_should_stop_is_valid (false),
        m_should_perform_action (true)
    {
    }
    
    StopInfoBreakpoint (Thread &thread, break_id_t break_id, bool should_stop) :
        StopInfo (thread, break_id),
        m_description(),
        m_should_stop (should_stop),
        m_should_stop_is_valid (true),
        m_should_perform_action (true)
    {
    }

    virtual ~StopInfoBreakpoint ()
    {
    }
    
    virtual StopReason
    GetStopReason () const
    {
        return eStopReasonBreakpoint;
    }

    virtual bool
    ShouldStop (Event *event_ptr)
    {
        if (!m_should_stop_is_valid)
        {
            // Only check once if we should stop at a breakpoint
            BreakpointSiteSP bp_site_sp (m_thread.GetProcess().GetBreakpointSiteList().FindByID (m_value));
            if (bp_site_sp)
            {
                StoppointCallbackContext context (event_ptr, 
                                                  &m_thread.GetProcess(), 
                                                  &m_thread, 
                                                  m_thread.GetStackFrameAtIndex(0).get(),
                                                  true);
                
                m_should_stop = bp_site_sp->ShouldStop (&context);
            }
            else
            {
                LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_PROCESS));

                if (log)
                    log->Printf ("Process::%s could not find breakpoint site id: %lld...", __FUNCTION__, m_value);

                m_should_stop = true;
            }
            m_should_stop_is_valid = true;
        }
        return m_should_stop;
    }
    
    virtual void
    PerformAction (Event *event_ptr)
    {
        if (!m_should_perform_action)
            return;
        m_should_perform_action = false;
        
        BreakpointSiteSP bp_site_sp (m_thread.GetProcess().GetBreakpointSiteList().FindByID (m_value));
        if (bp_site_sp)
        {
            size_t num_owners = bp_site_sp->GetNumberOfOwners();
            for (size_t j = 0; j < num_owners; j++)
            {
                // The breakpoint action is an asynchronous breakpoint callback.  If we ever need to have both
                // callbacks and actions on the same breakpoint, we'll have to split this into two.
                lldb::BreakpointLocationSP bp_loc_sp = bp_site_sp->GetOwnerAtIndex(j);
                StoppointCallbackContext context (event_ptr, 
                                                  &m_thread.GetProcess(), 
                                                  &m_thread, 
                                                  m_thread.GetStackFrameAtIndex(0).get(),
                                                  false);
                bp_loc_sp->InvokeCallback (&context);
            }
        }
        else
        {
            LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_PROCESS));

            if (log)
                log->Printf ("Process::%s could not find breakpoint site id: %lld...", __FUNCTION__, m_value);
        }
    }
        
    virtual bool
    ShouldNotify (Event *event_ptr)
    {
        BreakpointSiteSP bp_site_sp (m_thread.GetProcess().GetBreakpointSiteList().FindByID (m_value));
        if (bp_site_sp)
        {
            bool all_internal = true;

            for (uint32_t i = 0; i < bp_site_sp->GetNumberOfOwners(); i++)
            {
                if (!bp_site_sp->GetOwnerAtIndex(i)->GetBreakpoint().IsInternal())
                {
                    all_internal = false;
                    break;
                }
            }
            return all_internal == false;
        }
        return true;
    }

    virtual const char *
    GetDescription ()
    {
        if (m_description.empty())
        {
            BreakpointSiteSP bp_site_sp (m_thread.GetProcess().GetBreakpointSiteList().FindByID (m_value));
            if (bp_site_sp)
            {
                StreamString strm;
                strm.Printf("breakpoint ");
                bp_site_sp->GetDescription(&strm, eDescriptionLevelBrief);
                m_description.swap (strm.GetString());
            }
            else
            {
                StreamString strm;
                strm.Printf("breakpoint site %lli", m_value);
                m_description.swap (strm.GetString());
            }
        }
        return m_description.c_str();
    }

private:
    std::string m_description;
    bool m_should_stop;
    bool m_should_stop_is_valid;
    bool m_should_perform_action; // Since we are trying to preserve the "state" of the system even if we run functions
                                  // etc. behind the users backs, we need to make sure we only REALLY perform the action once.
};


//----------------------------------------------------------------------
// StopInfoWatchpoint
//----------------------------------------------------------------------

class StopInfoWatchpoint : public StopInfo
{
public:

    StopInfoWatchpoint (Thread &thread, break_id_t watch_id) :
        StopInfo (thread, watch_id),
        m_description()
    {
    }
    
    virtual ~StopInfoWatchpoint ()
    {
    }

    virtual StopReason
    GetStopReason () const
    {
        return eStopReasonWatchpoint;
    }

    virtual const char *
    GetDescription ()
    {
        if (m_description.empty())
        {
            StreamString strm;
            strm.Printf("watchpoint %lli", m_value);
            m_description.swap (strm.GetString());
        }
        return m_description.c_str();
    }



private:
    std::string m_description;
};



//----------------------------------------------------------------------
// StopInfoUnixSignal
//----------------------------------------------------------------------

class StopInfoUnixSignal : public StopInfo
{
public:

    StopInfoUnixSignal (Thread &thread, int signo) :
        StopInfo (thread, signo)
    {
    }
    
    virtual ~StopInfoUnixSignal ()
    {
    }


    virtual StopReason
    GetStopReason () const
    {
        return eStopReasonSignal;
    }

    virtual bool
    ShouldStop (Event *event_ptr)
    {
        return m_thread.GetProcess().GetUnixSignals().GetShouldStop (m_value);
    }
    
    
    // If should stop returns false, check if we should notify of this event
    virtual bool
    ShouldNotify (Event *event_ptr)
    {
        return m_thread.GetProcess().GetUnixSignals().GetShouldNotify (m_value);
    }

    
    virtual void
    WillResume (lldb::StateType resume_state)
    {
        if (m_thread.GetProcess().GetUnixSignals().GetShouldSuppress(m_value) == false)
            m_thread.SetResumeSignal(m_value);
    }

    virtual const char *
    GetDescription ()
    {
        if (m_description.empty())
        {
            StreamString strm;
            const char *signal_name = m_thread.GetProcess().GetUnixSignals().GetSignalAsCString (m_value);
            if (signal_name)
                strm.Printf("signal %s", signal_name);
            else
                strm.Printf("signal %lli", m_value);
            m_description.swap (strm.GetString());
        }
        return m_description.c_str();
    }
};

//----------------------------------------------------------------------
// StopInfoTrace
//----------------------------------------------------------------------

class StopInfoTrace : public StopInfo
{
public:

    StopInfoTrace (Thread &thread) :
        StopInfo (thread, LLDB_INVALID_UID)
    {
    }
    
    virtual ~StopInfoTrace ()
    {
    }
    
    virtual StopReason
    GetStopReason () const
    {
        return eStopReasonTrace;
    }

    virtual const char *
    GetDescription ()
    {
        if (m_description.empty())
        return "trace";
        else
            return m_description.c_str();
    }
};


//----------------------------------------------------------------------
// StopInfoException
//----------------------------------------------------------------------

class StopInfoException : public StopInfo
{
public:
    
    StopInfoException (Thread &thread, const char *description) :
        StopInfo (thread, LLDB_INVALID_UID)
    {
        if (description)
            SetDescription (description);
    }
    
    virtual 
    ~StopInfoException ()
    {
    }
    
    virtual StopReason
    GetStopReason () const
    {
        return eStopReasonException;
    }
    
    virtual const char *
    GetDescription ()
    {
        if (m_description.empty())
            return "exception";
        else
            return m_description.c_str();
    }
};


//----------------------------------------------------------------------
// StopInfoThreadPlan
//----------------------------------------------------------------------

class StopInfoThreadPlan : public StopInfo
{
public:

    StopInfoThreadPlan (ThreadPlanSP &plan_sp) :
        StopInfo (plan_sp->GetThread(), LLDB_INVALID_UID),
        m_plan_sp (plan_sp)
    {
    }
    
    virtual ~StopInfoThreadPlan ()
    {
    }

    virtual StopReason
    GetStopReason () const
    {
        return eStopReasonPlanComplete;
    }

    virtual const char *
    GetDescription ()
    {
        if (m_description.empty())
        {
            StreamString strm;            
            m_plan_sp->GetDescription (&strm, eDescriptionLevelBrief);
            m_description.swap (strm.GetString());
        }
        return m_description.c_str();
    }

private:
    ThreadPlanSP m_plan_sp;
};

StopInfoSP
StopInfo::CreateStopReasonWithBreakpointSiteID (Thread &thread, break_id_t break_id)
{
    return StopInfoSP (new StopInfoBreakpoint (thread, break_id));
}

StopInfoSP
StopInfo::CreateStopReasonWithBreakpointSiteID (Thread &thread, break_id_t break_id, bool should_stop)
{
    return StopInfoSP (new StopInfoBreakpoint (thread, break_id, should_stop));
}

StopInfoSP
StopInfo::CreateStopReasonWithWatchpointID (Thread &thread, break_id_t watch_id)
{
    return StopInfoSP (new StopInfoWatchpoint (thread, watch_id));
}

StopInfoSP
StopInfo::CreateStopReasonWithSignal (Thread &thread, int signo)
{
    return StopInfoSP (new StopInfoUnixSignal (thread, signo));
}

StopInfoSP
StopInfo::CreateStopReasonToTrace (Thread &thread)
{
    return StopInfoSP (new StopInfoTrace (thread));
}

StopInfoSP
StopInfo::CreateStopReasonWithPlan (ThreadPlanSP &plan_sp)
{
    return StopInfoSP (new StopInfoThreadPlan (plan_sp));
}

StopInfoSP
StopInfo::CreateStopReasonWithException (Thread &thread, const char *description)
{
    return StopInfoSP (new StopInfoException (thread, description));
}
