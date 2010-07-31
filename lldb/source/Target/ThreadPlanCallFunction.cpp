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
// Project includes
#include "lldb/lldb-private-log.h"
#include "lldb/Core/Address.h"
#include "lldb/Core/Log.h"
#include "lldb/Core/Stream.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/RegisterContext.h"
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
                                                bool discard_on_error) :
    ThreadPlan (ThreadPlan::eKindCallFunction, "Call function plan", thread, eVoteNoOpinion, eVoteNoOpinion),
    m_valid (false),
    m_stop_other_threads (stop_other_threads),
    m_arg_addr (arg),
    m_args (NULL),
    m_process (thread.GetProcess()),
    m_thread (thread)
{

    SetOkayToDiscard (discard_on_error);

    Process& process = thread.GetProcess();
    Target& target = process.GetTarget();
    const ABI *abi = process.GetABI();

    if (!abi)
        return;

    lldb::addr_t spBelowRedZone = thread.GetRegisterContext()->GetSP() - abi->GetRedZoneSize();
    
    SymbolContextList contexts;
    SymbolContext context;
    ModuleSP executableModuleSP (target.GetExecutableModule());

    if (!executableModuleSP ||
        !executableModuleSP->FindSymbolsWithNameAndType(ConstString ("start"), eSymbolTypeCode, contexts))
        return;
    
    contexts.GetContextAtIndex(0, context);
    
    m_start_addr = context.symbol->GetValue();
    lldb::addr_t StartLoadAddr = m_start_addr.GetLoadAddress(&process);

    if (!thread.SaveFrameZeroState(m_register_backup))
        return;

    m_function_addr = function;
    lldb::addr_t FunctionLoadAddr = m_function_addr.GetLoadAddress(&process);
        
    if (!abi->PrepareTrivialCall(thread, 
                                 spBelowRedZone, 
                                 FunctionLoadAddr, 
                                 StartLoadAddr, 
                                 m_arg_addr))
        return;
    
    m_valid = true;    
}

ThreadPlanCallFunction::ThreadPlanCallFunction (Thread &thread,
                                                Address &function,
                                                ValueList &args,
                                                bool stop_other_threads,
                                                bool discard_on_error) :
    ThreadPlan (ThreadPlan::eKindCallFunction, "Call function plan", thread, eVoteNoOpinion, eVoteNoOpinion),
    m_valid (false),
    m_stop_other_threads (stop_other_threads),
    m_arg_addr (0),
    m_args (&args),
    m_process (thread.GetProcess()),
    m_thread (thread)
{
    
    SetOkayToDiscard (discard_on_error);
    
    Process& process = thread.GetProcess();
    Target& target = process.GetTarget();
    const ABI *abi = process.GetABI();
    
    if(!abi)
        return;
    
    lldb::addr_t spBelowRedZone = thread.GetRegisterContext()->GetSP() - abi->GetRedZoneSize();
    
    SymbolContextList contexts;
    SymbolContext context;
    ModuleSP executableModuleSP (target.GetExecutableModule());
    
    if (!executableModuleSP ||
        !executableModuleSP->FindSymbolsWithNameAndType(ConstString ("start"), eSymbolTypeCode, contexts))
        return;
    
    contexts.GetContextAtIndex(0, context);
    
    m_start_addr = context.symbol->GetValue();
    lldb::addr_t StartLoadAddr = m_start_addr.GetLoadAddress(&process);
    
    if(!thread.SaveFrameZeroState(m_register_backup))
        return;
    
    m_function_addr = function;
    lldb::addr_t FunctionLoadAddr = m_function_addr.GetLoadAddress(&process);
    
    if (!abi->PrepareNormalCall(thread, 
                                spBelowRedZone, 
                                FunctionLoadAddr, 
                                StartLoadAddr, 
                                *m_args))
        return;
    
    m_valid = true;    
}

ThreadPlanCallFunction::~ThreadPlanCallFunction ()
{
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
            s->Printf("Thread plan to call 0x%llx with parsed arguments", m_function_addr.GetLoadAddress(&m_process), m_arg_addr);
        else
            s->Printf("Thread plan to call 0x%llx void * argument at: 0x%llx", m_function_addr.GetLoadAddress(&m_process), m_arg_addr);
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
    if (!m_subplan_sp)
        return false;
    else
        return m_subplan_sp->PlanExplainsStop();
}

bool
ThreadPlanCallFunction::ShouldStop (Event *event_ptr)
{
    if (PlanExplainsStop())
    {
        Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_STEP);
        
        if (log)
        {
            RegisterContext *reg_ctx = m_thread.GetRegisterContext();

            log->PutCString("Function completed.  Register state was:");

            for (uint32_t register_index = 0, num_registers = reg_ctx->GetRegisterCount();
                 register_index < num_registers;
                 ++register_index)
            {
                const char *register_name = reg_ctx->GetRegisterName(register_index);
                uint64_t register_value = reg_ctx->ReadRegisterAsUnsigned(register_index, LLDB_INVALID_ADDRESS);
                
                log->Printf("  %s = 0x%llx", register_name, register_value);
            }
        }
        
        m_thread.RestoreSaveFrameZero(m_register_backup);
        m_thread.ClearStackFrames();
        SetPlanComplete();
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
ThreadPlanCallFunction::RunState ()
{
    return eStateRunning;
}

void
ThreadPlanCallFunction::DidPush ()
{
    m_subplan_sp.reset(new ThreadPlanRunToAddress(m_thread, m_start_addr, m_stop_other_threads));
    
    m_thread.QueueThreadPlan(m_subplan_sp, false);

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
        Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_STEP);

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
