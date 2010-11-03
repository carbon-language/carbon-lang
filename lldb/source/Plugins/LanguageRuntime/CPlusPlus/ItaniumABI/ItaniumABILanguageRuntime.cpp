//===-- ItaniumABILanguageRuntime.cpp --------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "ItaniumABILanguageRuntime.h"

#include "lldb/Breakpoint/BreakpointLocation.h"
#include "lldb/Core/ConstString.h"
#include "lldb/Core/Error.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Core/Scalar.h"
#include "lldb/Symbol/ClangASTContext.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/RegisterContext.h"
#include "lldb/Target/StopInfo.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/Thread.h"

#include <vector>

using namespace lldb;
using namespace lldb_private;

static const char *pluginName = "ItaniumABILanguageRuntime";
static const char *pluginDesc = "Itanium ABI for the C++ language";
static const char *pluginShort = "language.itanium";

lldb::ValueObjectSP
ItaniumABILanguageRuntime::GetDynamicValue (ValueObjectSP in_value, ExecutionContextScope *exe_scope)
{
    ValueObjectSP ret_sp;
    return ret_sp;
}

bool
ItaniumABILanguageRuntime::IsVTableName (const char *name)
{
    if (name == NULL)
        return false;
        
    // Can we maybe ask Clang about this?
    if (strstr (name, "_vptr$") == name)
        return true;
    else
        return false;
}

//------------------------------------------------------------------
// Static Functions
//------------------------------------------------------------------
lldb_private::LanguageRuntime *
ItaniumABILanguageRuntime::CreateInstance (Process *process, lldb::LanguageType language)
{
    // FIXME: We have to check the process and make sure we actually know that this process supports
    // the Itanium ABI.
    if (language == eLanguageTypeC_plus_plus)
        return new ItaniumABILanguageRuntime (process);
    else
        return NULL;
}

void
ItaniumABILanguageRuntime::Initialize()
{
    PluginManager::RegisterPlugin (pluginName,
                                   pluginDesc,
                                   CreateInstance);    
}

void
ItaniumABILanguageRuntime::Terminate()
{
    PluginManager::UnregisterPlugin (CreateInstance);
}

//------------------------------------------------------------------
// PluginInterface protocol
//------------------------------------------------------------------
const char *
ItaniumABILanguageRuntime::GetPluginName()
{
    return pluginName;
}

const char *
ItaniumABILanguageRuntime::GetShortPluginName()
{
    return pluginShort;
}

uint32_t
ItaniumABILanguageRuntime::GetPluginVersion()
{
    return 1;
}

void
ItaniumABILanguageRuntime::GetPluginCommandHelp (const char *command, Stream *strm)
{
}

Error
ItaniumABILanguageRuntime::ExecutePluginCommand (Args &command, Stream *strm)
{
    Error error;
    error.SetErrorString("No plug-in command are currently supported.");
    return error;
}

Log *
ItaniumABILanguageRuntime::EnablePluginLogging (Stream *strm, Args &command)
{
    return NULL;
}

void
ItaniumABILanguageRuntime::SetExceptionBreakpoints ()
{
    if (!m_process)
        return;
    
    if (!m_cxx_exception_bp_sp)
        m_cxx_exception_bp_sp = m_process->GetTarget().CreateBreakpoint (NULL,
                                                                         "__cxa_throw",
                                                                         eFunctionNameTypeBase, 
                                                                         true);
    
    if (!m_cxx_exception_alloc_bp_sp)
        m_cxx_exception_alloc_bp_sp = m_process->GetTarget().CreateBreakpoint (NULL,
                                                                               "__cxa_allocate",
                                                                               eFunctionNameTypeBase,
                                                                               true);
}

void
ItaniumABILanguageRuntime::ClearExceptionBreakpoints ()
{
    if (!m_process)
        return;
    
    if (m_cxx_exception_bp_sp.get())
    {
        m_process->GetTarget().RemoveBreakpointByID(m_cxx_exception_bp_sp->GetID());
        m_cxx_exception_bp_sp.reset();
    }
    
    if (m_cxx_exception_alloc_bp_sp.get())
    {
        m_process->GetTarget().RemoveBreakpointByID(m_cxx_exception_alloc_bp_sp->GetID());
        m_cxx_exception_bp_sp.reset();
    }
}

bool
ItaniumABILanguageRuntime::ExceptionBreakpointsExplainStop (lldb::StopInfoSP stop_reason)
{
    if (!m_process)
        return false;
    
    if (!stop_reason || 
        stop_reason->GetStopReason() != eStopReasonBreakpoint)
        return false;
    
    uint64_t break_site_id = stop_reason->GetValue();
    lldb::BreakpointSiteSP bp_site_sp = m_process->GetBreakpointSiteList().FindByID(break_site_id);
    
    if (!bp_site_sp)
        return false;
    
    uint32_t num_owners = bp_site_sp->GetNumberOfOwners();
    
    bool        check_cxx_exception = false;
    break_id_t  cxx_exception_bid;
    
    bool        check_cxx_exception_alloc = false;
    break_id_t  cxx_exception_alloc_bid;
    
    if (m_cxx_exception_bp_sp)
    {
        check_cxx_exception = true;
        cxx_exception_bid = m_cxx_exception_bp_sp->GetID();
    }
    
    if (m_cxx_exception_alloc_bp_sp)
    {
        check_cxx_exception_alloc = true;
        cxx_exception_alloc_bid = m_cxx_exception_alloc_bp_sp->GetID();
    }
    
    for (uint32_t i = 0; i < num_owners; i++)
    {
        break_id_t bid = bp_site_sp->GetOwnerAtIndex(i)->GetBreakpoint().GetID();
        
        if ((check_cxx_exception        && (bid == cxx_exception_bid)) ||
            (check_cxx_exception_alloc  && (bid == cxx_exception_alloc_bid)))
            return true;
    }
    
    return false;
}
