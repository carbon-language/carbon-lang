//===-- ThreadPlanStepOverRange.cpp -----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Target/ThreadPlanStepOverRange.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes

#include "lldb/lldb-private-log.h"
#include "lldb/Core/Log.h"
#include "lldb/Core/Stream.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/RegisterContext.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/Thread.h"
#include "lldb/Target/ThreadPlanStepOut.h"
#include "lldb/Target/ThreadPlanStepThrough.h"

using namespace lldb_private;
using namespace lldb;


//----------------------------------------------------------------------
// ThreadPlanStepOverRange: Step through a stack range, either stepping over or into
// based on the value of \a type.
//----------------------------------------------------------------------

ThreadPlanStepOverRange::ThreadPlanStepOverRange
(
    Thread &thread,
    const AddressRange &range,
    const SymbolContext &addr_context,
    lldb::RunMode stop_others
) :
    ThreadPlanStepRange (ThreadPlan::eKindStepOverRange, "Step range stepping over", thread, range, addr_context, stop_others)
{
}

ThreadPlanStepOverRange::~ThreadPlanStepOverRange ()
{
}

void
ThreadPlanStepOverRange::GetDescription (Stream *s, lldb::DescriptionLevel level)
{
    if (level == lldb::eDescriptionLevelBrief)
        s->Printf("step over");
    else
    {
        s->Printf ("stepping through range (stepping over functions): ");
        DumpRanges(s);    
    }
}

bool
ThreadPlanStepOverRange::ShouldStop (Event *event_ptr)
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_STEP));

    if (log)
    {
        StreamString s;
        s.Address (m_thread.GetRegisterContext()->GetPC(), 
                   m_thread.CalculateTarget()->GetArchitecture().GetAddressByteSize());
        log->Printf("ThreadPlanStepOverRange reached %s.", s.GetData());
    }
    
    // If we're out of the range but in the same frame or in our caller's frame
    // then we should stop.
    // When stepping out we only step if we are forcing running one thread.
    bool stop_others;
    if (m_stop_others == lldb::eOnlyThisThread)
        stop_others = true;
    else 
        stop_others = false;

    ThreadPlan* new_plan = NULL;
    
    FrameComparison frame_order = CompareCurrentFrameToStartFrame();
    
    if (frame_order == eFrameCompareOlder)
    {
        // If we're in an older frame then we should stop.
        //
        // A caveat to this is if we think the frame is older but we're actually in a trampoline.
        // I'm going to make the assumption that you wouldn't RETURN to a trampoline.  So if we are
        // in a trampoline we think the frame is older because the trampoline confused the backtracer.
        // As below, we step through first, and then try to figure out how to get back out again.
        
        new_plan = m_thread.QueueThreadPlanForStepThrough (m_stack_id, false, stop_others);

        if (new_plan != NULL && log)
            log->Printf("Thought I stepped out, but in fact arrived at a trampoline.");
    }
    else if (frame_order == eFrameCompareYounger)
    {
        // Make sure we really are in a new frame.  Do that by unwinding and seeing if the
        // start function really is our start function...
        StackFrameSP older_frame_sp = m_thread.GetStackFrameAtIndex(1);
        
        // But if we can't even unwind one frame we should just get out of here & stop...
        if (older_frame_sp)
        {
            const SymbolContext &older_context = older_frame_sp->GetSymbolContext(eSymbolContextEverything);
            
            // Match as much as is specified in the m_addr_context:
            // This is a fairly loose sanity check.  Note, sometimes the target doesn't get filled
            // in so I left out the target check.  And sometimes the module comes in as the .o file from the
            // inlined range, so I left that out too...
            
            bool older_ctx_is_equivalent = false;
            if (m_addr_context.comp_unit)
            {
                if (m_addr_context.comp_unit == older_context.comp_unit)
                {
                    if (m_addr_context.function && m_addr_context.function == older_context.function)
                    {
                        if (m_addr_context.block && m_addr_context.block == older_context.block)
                        {
                            older_ctx_is_equivalent = true;
                            if (m_addr_context.line_entry.IsValid() && LineEntry::Compare(m_addr_context.line_entry, older_context.line_entry) != 0)
                            {
                                older_ctx_is_equivalent = false;
                            }
                        }
                    }
                }
            }
            else if (m_addr_context.symbol && m_addr_context.symbol == older_context.symbol)
            {
                older_ctx_is_equivalent = true;
            }
        
            if (older_ctx_is_equivalent)
            {
                new_plan = m_thread.QueueThreadPlanForStepOut (false, 
                                                           NULL, 
                                                           true, 
                                                           stop_others, 
                                                           eVoteNo, 
                                                           eVoteNoOpinion,
                                                           0);
            }
            else 
            {
                new_plan = m_thread.QueueThreadPlanForStepThrough (m_stack_id, false, stop_others);
                
            }
        }
    }
    else
    {
        // If we're still in the range, keep going.
        if (InRange())
        {
            SetNextBranchBreakpoint();
            return false;
        }


        if (!InSymbol())
        {
            // This one is a little tricky.  Sometimes we may be in a stub or something similar,
            // in which case we need to get out of there.  But if we are in a stub then it's 
            // likely going to be hard to get out from here.  It is probably easiest to step into the
            // stub, and then it will be straight-forward to step out.        
            new_plan = m_thread.QueueThreadPlanForStepThrough (m_stack_id, false, stop_others);
        }
    }

    // If we get to this point, we're not going to use a previously set "next branch" breakpoint, so delete it:
    ClearNextBranchBreakpoint();
    
    if (new_plan == NULL)
        m_no_more_plans = true;
    else
        m_no_more_plans = false;

    if (new_plan == NULL)
    {
        // For efficiencies sake, we know we're done here so we don't have to do this
        // calculation again in MischiefManaged.
        SetPlanComplete();
        return true;
    }
    else
        return false;
}

bool
ThreadPlanStepOverRange::PlanExplainsStop ()
{
    // For crashes, breakpoint hits, signals, etc, let the base plan (or some plan above us)
    // handle the stop.  That way the user can see the stop, step around, and then when they
    // are done, continue and have their step complete.  The exception is if we've hit our
    // "run to next branch" breakpoint.
    // Note, unlike the step in range plan, we don't mark ourselves complete if we hit an
    // unexplained breakpoint/crash.
    
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_STEP));
    StopInfoSP stop_info_sp = GetPrivateStopReason();
    if (stop_info_sp)
    {
        StopReason reason = stop_info_sp->GetStopReason();

        switch (reason)
        {
        case eStopReasonBreakpoint:
            if (NextRangeBreakpointExplainsStop(stop_info_sp))
                return true;
            else
                return false;
            break;
        case eStopReasonWatchpoint:
        case eStopReasonSignal:
        case eStopReasonException:
            if (log)
                log->PutCString ("ThreadPlanStepInRange got asked if it explains the stop for some reason other than step.");
            return false;
            break;
        default:
            break;
        }
    }
    return true;
}
