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
FuncUnwinders::GetUnwindPlanAtCallSite (int current_offset)
{
    if (m_unwind_at_call_site != NULL)
        return m_unwind_at_call_site;
    if (!m_range.GetBaseAddress().IsValid())
        return NULL;

    // We have cases (e.g. with _sigtramp on Mac OS X) where the hand-written eh_frame unwind info for a
    // function does not cover the entire range of the function and so the FDE only lists a subset of the
    // address range.  If we try to look up the unwind info by the starting address of the function 
    // (i.e. m_range.GetBaseAddress()) we may not find the eh_frame FDE.  We need to use the actual byte offset
    // into the function when looking it up.

    Address current_pc (m_range.GetBaseAddress ());
    if (current_offset != -1)
        current_pc.SetOffset (current_pc.GetOffset() + current_offset);

    DWARFCallFrameInfo *eh_frame = m_unwind_table.GetEHFrameInfo();
    UnwindPlan *up = NULL;
    if (eh_frame)
    {
        up = new UnwindPlan;
        if (!eh_frame->GetUnwindPlan (current_pc, *up))
        {
            delete up;
            return NULL;
        }
    }
    if (!up)
        return NULL;

    m_unwind_at_call_site = up;
    return m_unwind_at_call_site;
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

