//===-- ThreadPlanCallFunction.cpp ------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Target/ThreadPlanCallFunction.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
#include "llvm/Support/MachO.h"
// Project includes
#include "lldb/lldb-private-log.h"
#include "lldb/Breakpoint/Breakpoint.h"
#include "lldb/Breakpoint/BreakpointLocation.h"
#include "lldb/Core/Address.h"
#include "lldb/Core/Log.h"
#include "lldb/Core/Stream.h"
#include "lldb/Target/LanguageRuntime.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/RegisterContext.h"
#include "lldb/Target/StopInfo.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/Thread.h"
#include "lldb/Target/ThreadPlanRunToAddress.h"

using namespace lldb;
using namespace lldb_private;

//----------------------------------------------------------------------
// ThreadPlanCallFunction: Plan to call a single function
//----------------------------------------------------------------------

ThreadPlanCallFunction::ThreadPlanCallFunction (Thread &thread,
                                                Address &function,
                                                lldb::addr_t arg,
                                                bool stop_other_threads,
                                                bool discard_on_error,
                                                lldb::addr_t *this_arg,
                                                lldb::addr_t *cmd_arg) :
    ThreadPlan (ThreadPlan::eKindCallFunction, "Call function plan", thread, eVoteNoOpinion, eVoteNoOpinion),
    m_valid (false),
    m_stop_other_threads (stop_other_threads),
    m_arg_addr (arg),
    m_args (NULL),
    m_process (thread.GetProcess()),
    m_thread (thread),
    m_takedown_done (false),
    m_function_sp(NULL)
{
    SetOkayToDiscard (discard_on_error);

    Process& process = thread.GetProcess();
    Target& target = process.GetTarget();
    const ABI *abi = process.GetABI().get();
    
    if (!abi)
        return;
    
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_STEP));
    
    SetBreakpoints();
    
    m_function_sp = thread.GetRegisterContext()->GetSP() - abi->GetRedZoneSize();
    
    ModuleSP executableModuleSP (target.GetExecutableModule());

    if (!executableModuleSP)
    {
        log->Printf ("Can't execute code without an executable module.");
        return;
    }
    else
    {
        ObjectFile *objectFile = executableModuleSP->GetObjectFile();
        if (!objectFile)
        {
            log->Printf ("Could not find object file for module \"%s\".", 
                         executableModuleSP->GetFileSpec().GetFilename().AsCString());
            return;
        }
        m_start_addr = objectFile->GetEntryPointAddress();
        if (!m_start_addr.IsValid())
        {
            log->Printf ("Could not find entry point address for executable module \"%s\".", 
                         executableModuleSP->GetFileSpec().GetFilename().AsCString());
            return;
        }
    }
    
    lldb::addr_t StartLoadAddr = m_start_addr.GetLoadAddress(&target);
    
    // Checkpoint the thread state so we can restore it later.
    if (log && log->GetVerbose())
        ReportRegisterState ("About to checkpoint thread before function call.  Original register state was:");

    if (!thread.CheckpointThreadState (m_stored_thread_state))
    {
        if (log)
            log->Printf ("Setting up ThreadPlanCallFunction, failed to checkpoint thread state.");
        return;
    }
    // Now set the thread state to "no reason" so we don't run with whatever signal was outstanding...
    thread.SetStopInfoToNothing();
    
    m_function_addr = function;
    lldb::addr_t FunctionLoadAddr = m_function_addr.GetLoadAddress(&target);
        
    if (this_arg && cmd_arg)
    {
        if (!abi->PrepareTrivialCall (thread, 
                                      m_function_sp, 
                                      FunctionLoadAddr, 
                                      StartLoadAddr, 
                                      this_arg,
                                      cmd_arg,
                                      &m_arg_addr))
            return;
    }
    else if (this_arg)
    {
        if (!abi->PrepareTrivialCall (thread, 
                                      m_function_sp, 
                                      FunctionLoadAddr, 
                                      StartLoadAddr, 
                                      this_arg,
                                      &m_arg_addr,
                                      NULL))
            return;
    }
    else
    {
        if (!abi->PrepareTrivialCall (thread, 
                                      m_function_sp, 
                                      FunctionLoadAddr, 
                                      StartLoadAddr, 
                                      &m_arg_addr,
                                      NULL,
                                      NULL))
            return;
    }
    
    ReportRegisterState ("Function call was set up.  Register state was:");
    
    m_valid = true;    
}

ThreadPlanCallFunction::~ThreadPlanCallFunction ()
{
}

void
ThreadPlanCallFunction::ReportRegisterState (const char *message)
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_STEP));
    if (log)
    {
        RegisterContext *reg_ctx = m_thread.GetRegisterContext().get();

        log->PutCString(message);

        for (uint32_t register_index = 0, num_registers = reg_ctx->GetRegisterCount();
             register_index < num_registers;
             ++register_index)
        {
            const char *register_name = reg_ctx->GetRegisterName(register_index);
            uint64_t register_value = reg_ctx->ReadRegisterAsUnsigned(register_index, LLDB_INVALID_ADDRESS);
            
            log->Printf("  %s = 0x%llx", register_name, register_value);
        }
    }
}

void
ThreadPlanCallFunction::DoTakedown ()
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_STEP));
    if (!m_takedown_done)
    {
        if (log)
            log->Printf ("DoTakedown called for thread 0x%4.4x, m_valid: %d complete: %d.\n", m_thread.GetID(), m_valid, IsPlanComplete());
        m_takedown_done = true;
        m_thread.RestoreThreadStateFromCheckpoint(m_stored_thread_state);
        SetPlanComplete();
        ClearBreakpoints();
        if (log && log->GetVerbose())
            ReportRegisterState ("Restoring thread state after function call.  Restored register state:");

    }
    else
    {
        if (log)
            log->Printf ("DoTakedown called as no-op for thread 0x%4.4x, m_valid: %d complete: %d.\n", m_thread.GetID(), m_valid, IsPlanComplete());
    }
}

void
ThreadPlanCallFunction::WillPop ()
{
    DoTakedown();
}

void
ThreadPlanCallFunction::GetDescription (Stream *s, lldb::DescriptionLevel level)
{
    if (level == lldb::eDescriptionLevelBrief)
    {
        s->Printf("Function call thread plan");
    }
    else
    {
        if (m_args)
            s->Printf("Thread plan to call 0x%llx with parsed arguments", m_function_addr.GetLoadAddress(&m_process.GetTarget()), m_arg_addr);
        else
            s->Printf("Thread plan to call 0x%llx void * argument at: 0x%llx", m_function_addr.GetLoadAddress(&m_process.GetTarget()), m_arg_addr);
    }
}

bool
ThreadPlanCallFunction::ValidatePlan (Stream *error)
{
    if (!m_valid)
        return false;

    return true;
}

bool
ThreadPlanCallFunction::PlanExplainsStop ()
{    
    // If our subplan knows why we stopped, even if it's done (which would forward the question to us)
    // we answer yes.
    if(m_subplan_sp.get() != NULL && m_subplan_sp->PlanExplainsStop())
        return true;
    
    // Check if the breakpoint is one of ours.
    
    if (BreakpointsExplainStop())
        return true;
    
    // If we don't want to discard this plan, than any stop we don't understand should be propagated up the stack.
    if (!OkayToDiscard())
        return false;
            
    // Otherwise, check the case where we stopped for an internal breakpoint, in that case, continue on.
    // If it is not an internal breakpoint, consult OkayToDiscard.
    lldb::StopInfoSP stop_info_sp = GetPrivateStopReason();
    
    if (stop_info_sp && stop_info_sp->GetStopReason() == eStopReasonBreakpoint)
    {
        uint64_t break_site_id = stop_info_sp->GetValue();
        lldb::BreakpointSiteSP bp_site_sp = m_thread.GetProcess().GetBreakpointSiteList().FindByID(break_site_id);
        if (bp_site_sp)
        {
            uint32_t num_owners = bp_site_sp->GetNumberOfOwners();
            bool is_internal = true;
            for (uint32_t i = 0; i < num_owners; i++)
            {
                Breakpoint &bp = bp_site_sp->GetOwnerAtIndex(i)->GetBreakpoint();
                
                if (!bp.IsInternal())
                {
                    is_internal = false;
                    break;
                }
            }
            if (is_internal)
                return false;
        }
        
        return OkayToDiscard();
    }
    else
    {
        // If the subplan is running, any crashes are attributable to us.
        // If we want to discard the plan, then we say we explain the stop
        // but if we are going to be discarded, let whoever is above us
        // explain the stop.
        return ((m_subplan_sp.get() != NULL) && !OkayToDiscard());
    }
}

bool
ThreadPlanCallFunction::ShouldStop (Event *event_ptr)
{
    if (PlanExplainsStop())
    {
        ReportRegisterState ("Function completed.  Register state was:");
        
        DoTakedown();
        
        return true;
    }
    else
    {
        return false;
    }
}

bool
ThreadPlanCallFunction::StopOthers ()
{
    return m_stop_other_threads;
}

void
ThreadPlanCallFunction::SetStopOthers (bool new_value)
{
    if (m_subplan_sp)
    {
        ThreadPlanRunToAddress *address_plan = static_cast<ThreadPlanRunToAddress *>(m_subplan_sp.get());
        address_plan->SetStopOthers(new_value);
    }
    m_stop_other_threads = new_value;
}

StateType
ThreadPlanCallFunction::GetPlanRunState ()
{
    return eStateRunning;
}

void
ThreadPlanCallFunction::DidPush ()
{
//#define SINGLE_STEP_EXPRESSIONS
    
#ifndef SINGLE_STEP_EXPRESSIONS
    m_subplan_sp.reset(new ThreadPlanRunToAddress(m_thread, m_start_addr, m_stop_other_threads));
    
    m_thread.QueueThreadPlan(m_subplan_sp, false);
    m_subplan_sp->SetPrivate (true);
#endif
}

bool
ThreadPlanCallFunction::WillStop ()
{
    return true;
}

bool
ThreadPlanCallFunction::MischiefManaged ()
{
    if (IsPlanComplete())
    {
        LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_STEP));

        if (log)
            log->Printf("Completed call function plan.");

        ThreadPlan::MischiefManaged ();
        return true;
    }
    else
    {
        return false;
    }
}

void
ThreadPlanCallFunction::SetBreakpoints ()
{
    m_cxx_language_runtime = m_process.GetLanguageRuntime(eLanguageTypeC_plus_plus);
    m_objc_language_runtime = m_process.GetLanguageRuntime(eLanguageTypeObjC);
    
    if (m_cxx_language_runtime)
        m_cxx_language_runtime->SetExceptionBreakpoints();
    if (m_objc_language_runtime)
        m_objc_language_runtime->SetExceptionBreakpoints();
}

void
ThreadPlanCallFunction::ClearBreakpoints ()
{
    if (m_cxx_language_runtime)
        m_cxx_language_runtime->ClearExceptionBreakpoints();
    if (m_objc_language_runtime)
        m_objc_language_runtime->ClearExceptionBreakpoints();
}

bool
ThreadPlanCallFunction::BreakpointsExplainStop()
{
    lldb::StopInfoSP stop_info_sp = GetPrivateStopReason();
    
    if (m_cxx_language_runtime &&
        m_cxx_language_runtime->ExceptionBreakpointsExplainStop(stop_info_sp))
        return true;
    
    if (m_objc_language_runtime &&
        m_objc_language_runtime->ExceptionBreakpointsExplainStop(stop_info_sp))
        return true;
    
    return false;
}
