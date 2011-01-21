//===-- ThreadPlanStepInRange.cpp -------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Target/ThreadPlanStepInRange.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes

#include "lldb/lldb-private-log.h"
#include "lldb/Core/Log.h"
#include "lldb/Core/Stream.h"
#include "lldb/Symbol/Symbol.h"
#include "lldb/Symbol/Function.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/RegisterContext.h"
#include "lldb/Target/Thread.h"
#include "lldb/Target/ThreadPlanStepOut.h"
#include "lldb/Target/ThreadPlanStepThrough.h"
#include "lldb/Core/RegularExpression.h"

using namespace lldb;
using namespace lldb_private;

uint32_t ThreadPlanStepInRange::s_default_flag_values = ThreadPlanShouldStopHere::eAvoidNoDebug;

//----------------------------------------------------------------------
// ThreadPlanStepInRange: Step through a stack range, either stepping over or into
// based on the value of \a type.
//----------------------------------------------------------------------

ThreadPlanStepInRange::ThreadPlanStepInRange
(
    Thread &thread,
    const AddressRange &range,
    const SymbolContext &addr_context,
    lldb::RunMode stop_others
) :
    ThreadPlanStepRange (ThreadPlan::eKindStepInRange, "Step Range stepping in", thread, range, addr_context, stop_others),
    ThreadPlanShouldStopHere (this, ThreadPlanStepInRange::DefaultShouldStopHereCallback, NULL),
    m_step_past_prologue (true)
{
    SetFlagsToDefault ();
}

ThreadPlanStepInRange::~ThreadPlanStepInRange ()
{
}

void
ThreadPlanStepInRange::GetDescription (Stream *s, lldb::DescriptionLevel level)
{
    if (level == lldb::eDescriptionLevelBrief)
        s->Printf("step in");
    else
    {
        s->Printf ("Stepping through range (stepping into functions): ");
        m_address_range.Dump (s, &m_thread.GetProcess().GetTarget(), Address::DumpStyleLoadAddress);
    }
}

bool
ThreadPlanStepInRange::ShouldStop (Event *event_ptr)
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_STEP));
    m_no_more_plans = false;
    
    if (log)
    {
        StreamString s;
        s.Address (m_thread.GetRegisterContext()->GetPC(), m_thread.GetProcess().GetAddressByteSize());
        log->Printf("ThreadPlanStepInRange reached %s.", s.GetData());
    }

    // If we're still in the range, keep going.
    if (InRange())
        return false;

    ThreadPlan* new_plan = NULL;

    // Stepping through should be done stopping other threads in general, since we're setting a breakpoint and
    // continuing...
    
    bool stop_others;
    if (m_stop_others != lldb::eAllThreads)
        stop_others = true;
    else
        stop_others = false;
        
    if (FrameIsOlder())
    {
        // If we're in an older frame then we should stop.
        //
        // A caveat to this is if we think the frame is older but we're actually in a trampoline.
        // I'm going to make the assumption that you wouldn't RETURN to a trampoline.  So if we are
        // in a trampoline we think the frame is older because the trampoline confused the backtracer.
        new_plan = m_thread.QueueThreadPlanForStepThrough (false, stop_others);
        if (new_plan == NULL)
            return true;
        else if (log)
        {
            log->Printf("Thought I stepped out, but in fact arrived at a trampoline.");
        }

    }
    else if (!FrameIsYounger() && InSymbol())
    {
        // If we are not in a place we should step through, we're done.
        // One tricky bit here is that some stubs don't push a frame, so we have to check
        // both the case of a frame that is younger, or the same as this frame.  
        // However, if the frame is the same, and we are still in the symbol we started
        // in, the we don't need to do this.  This first check isn't strictly necessary,
        // but it is more efficient.
    
        SetPlanComplete();
        return true;
    }
    
    // We may have set the plan up above in the FrameIsOlder section:
    
    if (new_plan == NULL)
        new_plan = m_thread.QueueThreadPlanForStepThrough (false, stop_others);
    
    if (log)
    {
        if (new_plan != NULL)
            log->Printf ("Found a step through plan: %s", new_plan->GetName());
        else
            log->Printf ("No step through plan found.");
    }
    
    // If not, give the "should_stop" callback a chance to push a plan to get us out of here.
    // But only do that if we actually have stepped in.
    if (!new_plan && FrameIsYounger())
        new_plan = InvokeShouldStopHereCallback();

    // If we've stepped in and we are going to stop here, check to see if we were asked to
    // run past the prologue, and if so do that.
    
    if (new_plan == NULL && FrameIsYounger() && m_step_past_prologue)
    {
        lldb::StackFrameSP curr_frame = m_thread.GetStackFrameAtIndex(0);
        if (curr_frame)
        {
            size_t bytes_to_skip = 0;
            lldb::addr_t curr_addr = m_thread.GetRegisterContext()->GetPC();
            Address func_start_address;
            
            SymbolContext sc = curr_frame->GetSymbolContext (eSymbolContextFunction | eSymbolContextSymbol);
            
            if (sc.function)
            {
                func_start_address = sc.function->GetAddressRange().GetBaseAddress();
                if (curr_addr == func_start_address.GetLoadAddress(m_thread.CalculateTarget()))
                    bytes_to_skip = sc.function->GetPrologueByteSize();
            }
            else if (sc.symbol)
            {
                func_start_address = sc.symbol->GetValue();
                if (curr_addr == func_start_address.GetLoadAddress(m_thread.CalculateTarget()))
                    bytes_to_skip = sc.symbol->GetPrologueByteSize();
            }
            
            if (bytes_to_skip != 0)
            {
                func_start_address.Slide (bytes_to_skip);
                log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_STEP);
                if (log)
                    log->Printf ("Pushing past prologue ");
                    
                new_plan = m_thread.QueueThreadPlanForRunToAddress(false, func_start_address,true);
            }
        }
    }
    
     if (new_plan == NULL)
     {
        m_no_more_plans = true;
        SetPlanComplete();
        return true;
    }
    else
    {
        m_no_more_plans = false;
        return false;
    }
}

void
ThreadPlanStepInRange::SetFlagsToDefault ()
{
    GetFlags().Set(ThreadPlanStepInRange::s_default_flag_values);
}

void 
ThreadPlanStepInRange::SetAvoidRegexp(const char *name)
{
    if (m_avoid_regexp_ap.get() == NULL)
        m_avoid_regexp_ap.reset (new RegularExpression(name));

    m_avoid_regexp_ap->Compile (name);
}

void
ThreadPlanStepInRange::SetDefaultFlagValue (uint32_t new_value)
{
    // TODO: Should we test this for sanity?
    ThreadPlanStepInRange::s_default_flag_values = new_value;
}

bool
ThreadPlanStepInRange::FrameMatchesAvoidRegexp ()
{
    StackFrame *frame = GetThread().GetStackFrameAtIndex(0).get();

    RegularExpression *avoid_regexp_to_use;
    
    avoid_regexp_to_use = m_avoid_regexp_ap.get();
    if (avoid_regexp_to_use == NULL)
        avoid_regexp_to_use = GetThread().GetSymbolsToAvoidRegexp();
        
    if (avoid_regexp_to_use != NULL)
    {
        SymbolContext sc = frame->GetSymbolContext(eSymbolContextSymbol);
        if (sc.symbol != NULL)
        {
            const char *unnamed_symbol = "<UNKNOWN>";
            const char *sym_name = sc.symbol->GetMangled().GetName().AsCString(unnamed_symbol);
            if (strcmp (sym_name, unnamed_symbol) != 0)
               return avoid_regexp_to_use->Execute(sym_name);
        }
    }
    return false;
}

ThreadPlan *
ThreadPlanStepInRange::DefaultShouldStopHereCallback (ThreadPlan *current_plan, Flags &flags, void *baton)
{
    bool should_step_out = false;
    StackFrame *frame = current_plan->GetThread().GetStackFrameAtIndex(0).get();
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_STEP));

    if (flags.Test(eAvoidNoDebug))
    {
        if (!frame->HasDebugInformation())
        {
            if (log)
                log->Printf ("Stepping out of frame with no debug info");

            should_step_out = true;
        }
    }
    
    if (!should_step_out)
    {
        if (current_plan->GetKind() == eKindStepInRange)
        {
            ThreadPlanStepInRange *step_in_range_plan = static_cast<ThreadPlanStepInRange *> (current_plan);
            should_step_out = step_in_range_plan->FrameMatchesAvoidRegexp ();
        }
    }
    
    if (should_step_out)
    {
        // FIXME: Make sure the ThreadPlanForStepOut does the right thing with inlined functions.
        return current_plan->GetThread().QueueThreadPlanForStepOut (false, 
                                                                    NULL, 
                                                                    true, 
                                                                    current_plan->StopOthers(), 
                                                                    eVoteNo, 
                                                                    eVoteNoOpinion,
                                                                    0); // Frame index
    }

    return NULL;
}
