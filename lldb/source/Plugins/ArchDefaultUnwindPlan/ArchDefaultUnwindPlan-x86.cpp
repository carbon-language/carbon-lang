//===-- ArchDefaultUnwindPlan-x86.cpp --------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "ArchDefaultUnwindPlan-x86.h"
#include "lldb/Core/ArchSpec.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Target/ArchDefaultUnwindPlan.h"

using namespace lldb;
using namespace lldb_private;

lldb_private::ArchDefaultUnwindPlan *
ArchDefaultUnwindPlan_x86_64::CreateInstance (const lldb_private::ArchSpec &arch)
{
    if (arch.GetMachine () == llvm::Triple::x86_64)
        return new ArchDefaultUnwindPlan_x86_64 ();
    return NULL;
}

ArchDefaultUnwindPlan_x86_64::ArchDefaultUnwindPlan_x86_64() :
                lldb_private::ArchDefaultUnwindPlan(), 
                m_unwind_plan_sp (new UnwindPlan)
{ 
    UnwindPlan::Row row;
    UnwindPlan::Row::RegisterLocation regloc;

    m_unwind_plan_sp->SetRegisterKind (eRegisterKindGeneric);
    row.SetCFARegister (LLDB_REGNUM_GENERIC_FP);
    row.SetCFAOffset (2 * 8);
    row.SetOffset (0);

    regloc.SetAtCFAPlusOffset (2 * -8);
    row.SetRegisterInfo (LLDB_REGNUM_GENERIC_FP, regloc);
    regloc.SetAtCFAPlusOffset (1 * -8);
    row.SetRegisterInfo (LLDB_REGNUM_GENERIC_PC, regloc);
    regloc.SetIsCFAPlusOffset (0);
    row.SetRegisterInfo (LLDB_REGNUM_GENERIC_SP, regloc);

    m_unwind_plan_sp->AppendRow (row);
    m_unwind_plan_sp->SetSourceName ("x86_64 architectural default");
}

//------------------------------------------------------------------
// PluginInterface protocol in UnwindAssemblyParser_x86
//------------------------------------------------------------------

const char *
ArchDefaultUnwindPlan_x86_64::GetPluginName()
{
    return "ArchDefaultUnwindPlan_x86_64";
}

const char *
ArchDefaultUnwindPlan_x86_64::GetShortPluginName()
{
    return "lldb.arch-default-unwind-plan.x86-64";
}


uint32_t
ArchDefaultUnwindPlan_x86_64::GetPluginVersion()
{
    return 1;
}
void
ArchDefaultUnwindPlan_x86_64::Initialize()
{
    PluginManager::RegisterPlugin (GetPluginNameStatic(),
                                   GetPluginDescriptionStatic(),
                                   CreateInstance);
}

void
ArchDefaultUnwindPlan_x86_64::Terminate()
{
    PluginManager::UnregisterPlugin (CreateInstance);
}


const char *
ArchDefaultUnwindPlan_x86_64::GetPluginNameStatic()
{
    return "ArchDefaultUnwindPlan_x86_64";
}

const char *
ArchDefaultUnwindPlan_x86_64::GetPluginDescriptionStatic()
{
    return "x86_64 architecture default unwind plan assembly plugin.";
}

UnwindPlanSP
ArchDefaultUnwindPlan_x86_64::GetArchDefaultUnwindPlan (Thread& thread, Address current_pc)
{
    return m_unwind_plan_sp;
}



lldb_private::ArchDefaultUnwindPlan *
ArchDefaultUnwindPlan_i386::CreateInstance (const lldb_private::ArchSpec &arch)
{
    if (arch.GetMachine () == llvm::Triple::x86)
        return new ArchDefaultUnwindPlan_i386 ();
    return NULL;
}

ArchDefaultUnwindPlan_i386::ArchDefaultUnwindPlan_i386() :
                lldb_private::ArchDefaultUnwindPlan(), 
                m_unwind_plan_sp (new UnwindPlan)
{ 
    UnwindPlan::Row row;
    UnwindPlan::Row::RegisterLocation regloc;

    m_unwind_plan_sp->SetRegisterKind (eRegisterKindGeneric);
    row.SetCFARegister (LLDB_REGNUM_GENERIC_FP);
    row.SetCFAOffset (2 * 4);
    row.SetOffset (0);

    regloc.SetAtCFAPlusOffset (2 * -4);
    row.SetRegisterInfo (LLDB_REGNUM_GENERIC_FP, regloc);
    regloc.SetAtCFAPlusOffset (1 * -4);
    row.SetRegisterInfo (LLDB_REGNUM_GENERIC_PC, regloc);
    regloc.SetIsCFAPlusOffset (0);
    row.SetRegisterInfo (LLDB_REGNUM_GENERIC_SP, regloc);

    m_unwind_plan_sp->AppendRow (row);
    m_unwind_plan_sp->SetSourceName ("i386 architectural default");
}

//------------------------------------------------------------------
// PluginInterface protocol in UnwindAssemblyParser_x86
//------------------------------------------------------------------

const char *
ArchDefaultUnwindPlan_i386::GetPluginName()
{
    return "ArchDefaultUnwindPlan_i386";
}

const char *
ArchDefaultUnwindPlan_i386::GetShortPluginName()
{
    return "archdefaultunwindplan.x86";
}


uint32_t
ArchDefaultUnwindPlan_i386::GetPluginVersion()
{
    return 1;
}

void
ArchDefaultUnwindPlan_i386::Initialize()
{
    PluginManager::RegisterPlugin (GetPluginNameStatic(),
                                   GetPluginDescriptionStatic(),
                                   CreateInstance);
}

void
ArchDefaultUnwindPlan_i386::Terminate()
{
    PluginManager::UnregisterPlugin (CreateInstance);
}


const char *
ArchDefaultUnwindPlan_i386::GetPluginNameStatic()
{
    return "ArchDefaultUnwindPlan_i386";
}

const char *
ArchDefaultUnwindPlan_i386::GetPluginDescriptionStatic()
{
    return "i386 architecture default unwind plan assembly plugin.";
}

UnwindPlanSP
ArchDefaultUnwindPlan_i386::GetArchDefaultUnwindPlan (Thread& thread, Address current_pc)
{
    return m_unwind_plan_sp;
}

