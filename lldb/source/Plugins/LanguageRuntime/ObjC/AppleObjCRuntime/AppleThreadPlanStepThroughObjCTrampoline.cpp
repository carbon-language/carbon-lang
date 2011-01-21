//===-- AppleThreadPlanStepThroughObjCTrampoline.cpp --------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "AppleThreadPlanStepThroughObjCTrampoline.h"
#include "AppleObjCTrampolineHandler.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/Thread.h"
#include "lldb/Expression/ClangExpression.h"
#include "lldb/Expression/ClangFunction.h"
#include "lldb/Target/ExecutionContext.h"
#include "lldb/Target/ObjCLanguageRuntime.h"
#include "lldb/Target/ThreadPlanRunToAddress.h"
#include "lldb/Target/ThreadPlanStepOut.h"
#include "lldb/Core/Log.h"

using namespace lldb_private;

//----------------------------------------------------------------------
// ThreadPlanStepThroughObjCTrampoline constructor
//----------------------------------------------------------------------
AppleThreadPlanStepThroughObjCTrampoline::AppleThreadPlanStepThroughObjCTrampoline(
        Thread &thread, 
        AppleObjCTrampolineHandler *trampoline_handler, 
        lldb::addr_t args_addr, 
        lldb::addr_t object_addr,
        lldb::addr_t isa_addr,
        lldb::addr_t sel_addr,
        bool stop_others) :
    ThreadPlan (ThreadPlan::eKindGeneric, "MacOSX Step through ObjC Trampoline", thread, 
        lldb::eVoteNoOpinion, lldb::eVoteNoOpinion),
    m_trampoline_handler (trampoline_handler),
    m_args_addr (args_addr),
    m_object_addr (object_addr),
    m_isa_addr(isa_addr),
    m_sel_addr(sel_addr),
    m_impl_function (trampoline_handler->GetLookupImplementationWrapperFunction()),
    m_stop_others (stop_others)
{
    
}

//----------------------------------------------------------------------
// Destructor
//----------------------------------------------------------------------
AppleThreadPlanStepThroughObjCTrampoline::~AppleThreadPlanStepThroughObjCTrampoline()
{
}

void
AppleThreadPlanStepThroughObjCTrampoline::DidPush ()
{
    StreamString errors;
    ExecutionContext exc_context;
    m_thread.CalculateExecutionContext(exc_context);
    m_func_sp.reset(m_impl_function->GetThreadPlanToCallFunction (exc_context, m_args_addr, errors, m_stop_others));
    m_func_sp->SetPrivate(true);
    m_thread.QueueThreadPlan (m_func_sp, false);
}

void
AppleThreadPlanStepThroughObjCTrampoline::GetDescription (Stream *s,
                lldb::DescriptionLevel level)
{
    if (level == lldb::eDescriptionLevelBrief)
        s->Printf("Step through ObjC trampoline");
    else
    {
        s->Printf ("Stepping to implementation of ObjC method - obj: 0x%llx, isa: 0x%llx, sel: 0x%llx",
        m_object_addr, m_isa_addr, m_sel_addr);
    }
}
                
bool
AppleThreadPlanStepThroughObjCTrampoline::ValidatePlan (Stream *error)
{
    return true;
}

bool
AppleThreadPlanStepThroughObjCTrampoline::PlanExplainsStop ()
{
    // This plan should actually never stop when it is on the top of the plan
    // stack, since it does all it's running in client plans.
    return false;
}

lldb::StateType
AppleThreadPlanStepThroughObjCTrampoline::GetPlanRunState ()
{
    return eStateRunning;
}

bool
AppleThreadPlanStepThroughObjCTrampoline::ShouldStop (Event *event_ptr)
{
    if (m_func_sp.get() == NULL || m_thread.IsThreadPlanDone(m_func_sp.get()))
    {
        m_func_sp.reset();
        if (!m_run_to_sp) 
        {
            Value target_addr_value;
            ExecutionContext exc_context;
            m_thread.CalculateExecutionContext(exc_context);
            m_impl_function->FetchFunctionResults (exc_context, m_args_addr, target_addr_value);
            m_impl_function->DeallocateFunctionResults(exc_context, m_args_addr);
            lldb::addr_t target_addr = target_addr_value.GetScalar().ULongLong();
            Address target_address(NULL, target_addr);
            LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_STEP));
            if (target_addr == 0)
            {
                if (log)
                    log->Printf("Got target implementation of 0x0, stopping.");
                SetPlanComplete();
                return true;
            }
            if (m_trampoline_handler->AddrIsMsgForward(target_addr))
            {
                if (log)
                    log->Printf ("Implementation lookup returned msgForward function: 0x%llx, stopping.", target_addr);

                SymbolContext sc = m_thread.GetStackFrameAtIndex(0)->GetSymbolContext(eSymbolContextEverything);
                m_run_to_sp.reset(new ThreadPlanStepOut (m_thread, 
                                                         &sc, 
                                                         true, 
                                                         m_stop_others, 
                                                         eVoteNoOpinion, 
                                                         eVoteNoOpinion,
                                                         0));
                m_thread.QueueThreadPlan(m_run_to_sp, false);
                m_run_to_sp->SetPrivate(true);
                return false;
            }
            
            if (log)
                log->Printf("Running to ObjC method implementation: 0x%llx", target_addr);
            
            ObjCLanguageRuntime *objc_runtime = GetThread().GetProcess().GetObjCLanguageRuntime();
            assert (objc_runtime != NULL);
            objc_runtime->AddToMethodCache (m_isa_addr, m_sel_addr, target_addr);
            if (log)
                log->Printf("Adding {0x%llx, 0x%llx} = 0x%llx to cache.", m_isa_addr, m_sel_addr, target_addr);

            // Extract the target address from the value:
            
            m_run_to_sp.reset(new ThreadPlanRunToAddress(m_thread, target_address, m_stop_others));
            m_thread.QueueThreadPlan(m_run_to_sp, false);
            m_run_to_sp->SetPrivate(true);
            return false;
        }
        else if (m_thread.IsThreadPlanDone(m_run_to_sp.get()))
        {
            SetPlanComplete();
            return true;
        }
    }
    return false;
}

// The base class MischiefManaged does some cleanup - so you have to call it
// in your MischiefManaged derived class.
bool
AppleThreadPlanStepThroughObjCTrampoline::MischiefManaged ()
{
    if (IsPlanComplete())
        return true;
    else
        return false;
}

bool
AppleThreadPlanStepThroughObjCTrampoline::WillStop()
{
    return true;
}
