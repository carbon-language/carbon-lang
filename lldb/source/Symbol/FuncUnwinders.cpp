//===-- FuncUnwinders.cpp ----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/lldb-private.h"
#include "lldb/Symbol/FuncUnwinders.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Symbol/UnwindPlan.h"
#include "lldb/Core/AddressRange.h"
#include "lldb/Core/Address.h"
#include "lldb/Symbol/UnwindTable.h"
#include "lldb/Utility/UnwindAssemblyProfiler.h"
#include "lldb/Utility/ArchDefaultUnwindPlan.h"
#include "lldb/Symbol/DWARFCallFrameInfo.h"
#include "lldb/Target/Thread.h"
#include "lldb/Target/Target.h"

using namespace lldb;
using namespace lldb_private;


FuncUnwinders::FuncUnwinders (UnwindTable& unwind_table, UnwindAssemblyProfiler *assembly_profiler, AddressRange range) : 
        m_unwind_table(unwind_table), 
        m_assembly_profiler(assembly_profiler), 
        m_range(range), 
        m_unwind_at_call_site(NULL), 
        m_unwind_at_non_call_site(NULL), 
        m_fast_unwind(NULL), 
        m_arch_default_unwind(NULL), 
        m_first_non_prologue_insn() { }

FuncUnwinders::~FuncUnwinders () 
{ 
  if (m_unwind_at_call_site)
      delete m_unwind_at_call_site;
  if (m_unwind_at_non_call_site)
      delete m_unwind_at_non_call_site;
  if (m_fast_unwind)
      delete m_fast_unwind;
  if (m_arch_default_unwind)
      delete m_arch_default_unwind;
}

UnwindPlan*
FuncUnwinders::GetUnwindPlanAtCallSite ()
{
    if (m_unwind_at_call_site != NULL)
        return m_unwind_at_call_site;
    if (!m_range.GetBaseAddress().IsValid())
        return NULL;

    DWARFCallFrameInfo *eh_frame = m_unwind_table.GetEHFrameInfo();
    if (eh_frame)
    {
        UnwindPlan *up = new UnwindPlan;
        if (eh_frame->GetUnwindPlan (m_range.GetBaseAddress (), *up) == true)
        {
            m_unwind_at_call_site = up;
            return m_unwind_at_call_site;
        }
    }
    return NULL;
}

UnwindPlan*
FuncUnwinders::GetUnwindPlanAtNonCallSite (Thread& thread)
{
    if (m_unwind_at_non_call_site != NULL)
        return m_unwind_at_non_call_site;
    UnwindPlan *up = new UnwindPlan;
    if (!m_assembly_profiler->GetNonCallSiteUnwindPlanFromAssembly (m_range, thread, *up))
    {
        delete up;
        return NULL;
    }
    m_unwind_at_non_call_site = up;
    return m_unwind_at_non_call_site;
}

UnwindPlan*
FuncUnwinders::GetUnwindPlanFastUnwind (Thread& thread)
{
    if (m_fast_unwind != NULL)
        return m_fast_unwind;
    UnwindPlan *up = new UnwindPlan;
    if (!m_assembly_profiler->GetFastUnwindPlan (m_range, thread, *up))
    {
        delete up;
        return NULL;
    }
    m_fast_unwind = up;
    return m_fast_unwind;
}

UnwindPlan*
FuncUnwinders::GetUnwindPlanArchitectureDefault (Thread& thread)
{
    if (m_arch_default_unwind != NULL)
        return m_arch_default_unwind;

    Address current_pc;
    Target *target = thread.CalculateTarget();
    if (target)
    {
        ArchSpec arch = target->GetArchitecture ();
        ArchDefaultUnwindPlan *arch_default = ArchDefaultUnwindPlan::FindPlugin (arch);
        if (arch_default)
        {
            m_arch_default_unwind = arch_default->GetArchDefaultUnwindPlan (thread, current_pc);
        }
    }

    return m_arch_default_unwind;
}

Address&
FuncUnwinders::GetFirstNonPrologueInsn (Target& target)
{
    if (m_first_non_prologue_insn.IsValid())
        return m_first_non_prologue_insn;
    m_assembly_profiler->FirstNonPrologueInsn (m_range, target, NULL, m_first_non_prologue_insn);
    return m_first_non_prologue_insn;
}

const Address&
FuncUnwinders::GetFunctionStartAddress () const
{
    return m_range.GetBaseAddress();
}

