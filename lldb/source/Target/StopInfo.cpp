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
#include "lldb/Core/Debugger.h"
#include "lldb/Core/StreamString.h"
#include "lldb/Expression/ClangUserExpression.h"
#include "lldb/Target/Target.h"
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

lldb::StateType
StopInfo::GetPrivateState ()
{
    return m_thread.GetProcess().GetPrivateState();
}

//----------------------------------------------------------------------
// StopInfoBreakpoint
//----------------------------------------------------------------------

namespace lldb_private
{
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
        
        LogSP log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_BREAKPOINTS);
        // We're going to calculate whether we should stop or not in some way during the course of
        // this code.  So set the valid flag here.  Also by default we're going to stop, so 
        // set that here too.
        // m_should_stop_is_valid = true;
        m_should_stop = true;
        
        BreakpointSiteSP bp_site_sp (m_thread.GetProcess().GetBreakpointSiteList().FindByID (m_value));
        if (bp_site_sp)
        {
            size_t num_owners = bp_site_sp->GetNumberOfOwners();
            
            // We only continue from the callbacks if ALL the callbacks want us to continue.  
            // However we want to run all the callbacks, except of course if one of them actually
            // resumes the target.
            // So we use stop_requested to track what we're were asked to do.
            bool stop_requested = true;
            for (size_t j = 0; j < num_owners; j++)
            {
                lldb::BreakpointLocationSP bp_loc_sp = bp_site_sp->GetOwnerAtIndex(j);
                StoppointCallbackContext context (event_ptr, 
                                                  &m_thread.GetProcess(), 
                                                  &m_thread, 
                                                  m_thread.GetStackFrameAtIndex(0).get(),
                                                  false);
                stop_requested = bp_loc_sp->InvokeCallback (&context);
                // Also make sure that the callback hasn't continued the target.  
                // If it did, when we'll set m_should_start to false and get out of here.
                if (GetPrivateState() == eStateRunning)
                    m_should_stop = false;
            }
            
            if (m_should_stop && !stop_requested)
            {
                m_should_stop_is_valid = true;
                m_should_stop = false;
            }

            // Okay, so now if all the callbacks say we should stop, let's try the Conditions:
            if (m_should_stop)
            {
                size_t num_owners = bp_site_sp->GetNumberOfOwners();
                for (size_t j = 0; j < num_owners; j++)
                {
                    lldb::BreakpointLocationSP bp_loc_sp = bp_site_sp->GetOwnerAtIndex(j);
                    if (bp_loc_sp->GetConditionText() != NULL)
                    {
                        // We need to make sure the user sees any parse errors in their condition, so we'll hook the
                        // constructor errors up to the debugger's Async I/O.
                        
                        StoppointCallbackContext context (event_ptr, 
                                                          &m_thread.GetProcess(), 
                                                          &m_thread, 
                                                          m_thread.GetStackFrameAtIndex(0).get(),
                                                          false);
                        ValueObjectSP result_valobj_sp;
                        
                        ExecutionResults result_code;
                        ValueObjectSP result_value_sp;
                        const bool discard_on_error = true;
                        Error error;
                        result_code = ClangUserExpression::EvaluateWithError (context.exe_ctx,
                                                                              eExecutionPolicyAlways,
                                                                              discard_on_error,
                                                                              bp_loc_sp->GetConditionText(),
                                                                              NULL,
                                                                              result_value_sp,
                                                                              error);
                        if (result_code == eExecutionCompleted)
                        {
                            if (result_value_sp)
                            {
                                Scalar scalar_value;
                                if (result_value_sp->ResolveValue (scalar_value))
                                {
                                    if (scalar_value.ULongLong(1) == 0)
                                        m_should_stop = false;
                                    else
                                        m_should_stop = true;
                                    if (log)
                                        log->Printf("Condition successfully evaluated, result is %s.\n", 
                                                    m_should_stop ? "true" : "false");
                                }
                                else
                                {
                                    m_should_stop = true;
                                    if (log)
                                        log->Printf("Failed to get an integer result from the expression.");
                                }
                            }
                        }
                        else
                        {
                            Debugger &debugger = context.exe_ctx.target->GetDebugger();
                            StreamSP error_sp = debugger.GetAsyncErrorStream ();
                            error_sp->Printf ("Stopped due to an error evaluating condition of breakpoint ");
                            bp_loc_sp->GetDescription (error_sp.get(), eDescriptionLevelBrief);
                            error_sp->Printf (": \"%s\"", 
                                              bp_loc_sp->GetConditionText());
                            error_sp->EOL();
                            const char *err_str = error.AsCString("<Unknown Error>");
                            if (log)
                                log->Printf("Error evaluating condition: \"%s\"\n", err_str);
                            
                            error_sp->PutCString (err_str);
                            error_sp->EOL();                       
                            error_sp->Flush();
                            // If the condition fails to be parsed or run, we should stop.
                            m_should_stop = true;
                        }
                    }
                                            
                    // If any condition says we should stop, then we're going to stop, so we don't need
                    // to evaluate the others.
                    if (m_should_stop)
                        break;
                }
            }
        }
        else
        {
            LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_PROCESS));

            if (log)
                log->Printf ("Process::%s could not find breakpoint site id: %lld...", __FUNCTION__, m_value);
        }
        if (log)
            log->Printf ("Process::%s returning from action with m_should_stop: %d.", __FUNCTION__, m_should_stop);
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
} // namespace lldb_private

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
