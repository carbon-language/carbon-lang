//===-- ThreadPlanStepThroughObjCTrampoline.cpp --------------------------*- C++ -*-===//
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
#include "ThreadPlanStepThroughObjCTrampoline.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/Thread.h"
#include "lldb/Expression/ClangExpression.h"
#include "lldb/Expression/ClangFunction.h"
#include "lldb/Target/ExecutionContext.h"
#include "lldb/Target/ThreadPlanRunToAddress.h"
#include "lldb/Core/Log.h"

using namespace lldb_private;

//----------------------------------------------------------------------
// ThreadPlanStepThroughObjCTrampoline constructor
//----------------------------------------------------------------------
ThreadPlanStepThroughObjCTrampoline::ThreadPlanStepThroughObjCTrampoline(
        Thread &thread, 
        ObjCTrampolineHandler *trampoline_handler, 
        lldb::addr_t args_addr, 
        lldb::addr_t object_ptr, 
        lldb::addr_t class_ptr, 
        lldb::addr_t sel_ptr, 
        bool stop_others) :
    ThreadPlan (ThreadPlan::eKindGeneric, "MacOSX Step through ObjC Trampoline", thread, eVoteNoOpinion, eVoteNoOpinion),
    m_objc_trampoline_handler (trampoline_handler),
    m_impl_function (trampoline_handler->GetLookupImplementationWrapperFunction()),
    m_args_addr (args_addr),
    m_object_ptr (object_ptr),
    m_class_ptr (class_ptr),
    m_sel_ptr (sel_ptr),
    m_stop_others (stop_others)
{
    
}

//----------------------------------------------------------------------
// Destructor
//----------------------------------------------------------------------
ThreadPlanStepThroughObjCTrampoline::~ThreadPlanStepThroughObjCTrampoline()
{
}

void
ThreadPlanStepThroughObjCTrampoline::DidPush ()
{
    StreamString errors;
    ExecutionContext exc_context;
    m_thread.Calculate(exc_context);
    m_func_sp.reset(m_impl_function->GetThreadPlanToCallFunction (exc_context, m_args_addr, errors, m_stop_others));
    m_func_sp->SetPrivate(true);
    m_thread.QueueThreadPlan (m_func_sp, false);
}

void
ThreadPlanStepThroughObjCTrampoline::GetDescription (Stream *s,
                lldb::DescriptionLevel level)
{
    if (level == lldb::eDescriptionLevelBrief)
        s->Printf("Step through ObjC trampoline");
    else
    {
        s->Printf ("Stepping to implementation of ObjC method - obj: 0x%llx class: 0x%llx selector: 0x%llx",
        m_object_ptr, m_class_ptr, m_sel_ptr);
    }
}
                
bool
ThreadPlanStepThroughObjCTrampoline::ValidatePlan (Stream *error)
{
    return true;
}

bool
ThreadPlanStepThroughObjCTrampoline::PlanExplainsStop ()
{
    // This plan should actually never stop when it is on the top of the plan
    // stack, since it does all it's running in client plans.
    return false;
}

lldb::StateType
ThreadPlanStepThroughObjCTrampoline::RunState ()
{
    return eStateRunning;
}

bool
ThreadPlanStepThroughObjCTrampoline::ShouldStop (Event *event_ptr)
{
    if (m_func_sp.get() == NULL || m_thread.IsThreadPlanDone(m_func_sp.get()))
    {
        m_func_sp.reset();
        if (!m_run_to_sp) 
        {
            Value target_addr_value;
            ExecutionContext exc_context;
            m_thread.Calculate(exc_context);
            m_impl_function->FetchFunctionResults (exc_context, m_args_addr, target_addr_value);
            m_impl_function->DeallocateFunctionResults(exc_context, m_args_addr);
            lldb::addr_t target_addr = target_addr_value.GetScalar().ULongLong();
            Address target_address(NULL, target_addr);
            Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_STEP);
            if (log)
                log->Printf("Running to ObjC method implementation: 0x%llx", target_addr);
            
            m_objc_trampoline_handler->AddToCache (m_class_ptr, m_sel_ptr, target_addr);

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
ThreadPlanStepThroughObjCTrampoline::MischiefManaged ()
{
    if (IsPlanComplete())
        return true;
    else
        return false;
}

bool
ThreadPlanStepThroughObjCTrampoline::WillStop()
{
    return true;
}
