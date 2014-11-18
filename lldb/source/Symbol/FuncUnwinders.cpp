//===-- FuncUnwinders.cpp ----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Core/AddressRange.h"
#include "lldb/Core/Address.h"
#include "lldb/Symbol/FuncUnwinders.h"
#include "lldb/Symbol/DWARFCallFrameInfo.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Symbol/UnwindPlan.h"
#include "lldb/Symbol/UnwindTable.h"
#include "lldb/Target/ABI.h"
#include "lldb/Target/ExecutionContext.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/Thread.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/UnwindAssembly.h"

using namespace lldb;
using namespace lldb_private;


FuncUnwinders::FuncUnwinders (UnwindTable& unwind_table, AddressRange range) : 
    m_unwind_table (unwind_table), 
    m_range (range), 
    m_mutex (Mutex::eMutexTypeRecursive),
    m_unwind_plan_call_site_sp (), 
    m_unwind_plan_non_call_site_sp (), 
    m_unwind_plan_fast_sp (), 
    m_unwind_plan_arch_default_sp (), 
    m_tried_unwind_at_call_site (false),
    m_tried_unwind_at_non_call_site (false),
    m_tried_unwind_fast (false),
    m_tried_unwind_arch_default (false),
    m_tried_unwind_arch_default_at_func_entry (false),
    m_first_non_prologue_insn ()
{
}

FuncUnwinders::~FuncUnwinders ()
{ 
}

UnwindPlanSP
FuncUnwinders::GetUnwindPlanAtCallSite (int current_offset)
{
    Mutex::Locker locker (m_mutex);
    if (m_tried_unwind_at_call_site == false && m_unwind_plan_call_site_sp.get() == nullptr)
    {
        m_tried_unwind_at_call_site = true;
        // We have cases (e.g. with _sigtramp on Mac OS X) where the hand-written eh_frame unwind info for a
        // function does not cover the entire range of the function and so the FDE only lists a subset of the
        // address range.  If we try to look up the unwind info by the starting address of the function 
        // (i.e. m_range.GetBaseAddress()) we may not find the eh_frame FDE.  We need to use the actual byte offset
        // into the function when looking it up.

        if (m_range.GetBaseAddress().IsValid())
        {
            Address current_pc (m_range.GetBaseAddress ());
            if (current_offset != -1)
                current_pc.SetOffset (current_pc.GetOffset() + current_offset);

            DWARFCallFrameInfo *eh_frame = m_unwind_table.GetEHFrameInfo();
            if (eh_frame)
            {
                m_unwind_plan_call_site_sp.reset (new UnwindPlan (lldb::eRegisterKindGeneric));
                if (!eh_frame->GetUnwindPlan (current_pc, *m_unwind_plan_call_site_sp))
                    m_unwind_plan_call_site_sp.reset();
            }
        }
    }
    return m_unwind_plan_call_site_sp;
}

UnwindPlanSP
FuncUnwinders::GetUnwindPlanAtNonCallSite (Target& target, Thread& thread, int current_offset)
{
    Mutex::Locker locker (m_mutex);
    if (m_tried_unwind_at_non_call_site == false && m_unwind_plan_non_call_site_sp.get() == nullptr)
    {
        UnwindAssemblySP assembly_profiler_sp (GetUnwindAssemblyProfiler());
        if (assembly_profiler_sp)
        {
            if (target.GetArchitecture().GetCore() == ArchSpec::eCore_x86_32_i386
                || target.GetArchitecture().GetCore() == ArchSpec::eCore_x86_64_x86_64
                || target.GetArchitecture().GetCore() == ArchSpec::eCore_x86_64_x86_64h)
            {
                // For 0th frame on i386 & x86_64, we fetch eh_frame and try using assembly profiler
                // to augment it into asynchronous unwind table.
                GetUnwindPlanAtCallSite(current_offset);
                if (m_unwind_plan_call_site_sp) 
                {
                    UnwindPlan* plan = new UnwindPlan (*m_unwind_plan_call_site_sp);
                    if (assembly_profiler_sp->AugmentUnwindPlanFromCallSite (m_range, thread, *plan)) 
                    {
                        m_unwind_plan_non_call_site_sp.reset (plan);
                        return m_unwind_plan_non_call_site_sp;
                    }
                }
            }

            m_unwind_plan_non_call_site_sp.reset (new UnwindPlan (lldb::eRegisterKindGeneric));
            if (!assembly_profiler_sp->GetNonCallSiteUnwindPlanFromAssembly (m_range, thread, *m_unwind_plan_non_call_site_sp))
                m_unwind_plan_non_call_site_sp.reset();
        }
    }
    return m_unwind_plan_non_call_site_sp;
}

UnwindPlanSP
FuncUnwinders::GetUnwindPlanFastUnwind (Thread& thread)
{
    Mutex::Locker locker (m_mutex);
    if (m_tried_unwind_fast == false && m_unwind_plan_fast_sp.get() == nullptr)
    {
        m_tried_unwind_fast = true;
        UnwindAssemblySP assembly_profiler_sp (GetUnwindAssemblyProfiler());
        if (assembly_profiler_sp)
        {
            m_unwind_plan_fast_sp.reset (new UnwindPlan (lldb::eRegisterKindGeneric));
            if (!assembly_profiler_sp->GetFastUnwindPlan (m_range, thread, *m_unwind_plan_fast_sp))
                m_unwind_plan_fast_sp.reset();
        }
    }
    return m_unwind_plan_fast_sp;
}

UnwindPlanSP
FuncUnwinders::GetUnwindPlanArchitectureDefault (Thread& thread)
{
    Mutex::Locker locker (m_mutex);
    if (m_tried_unwind_arch_default == false && m_unwind_plan_arch_default_sp.get() == nullptr)
    {
        m_tried_unwind_arch_default = true;
        Address current_pc;
        ProcessSP process_sp (thread.CalculateProcess());
        if (process_sp)
        {
            ABI *abi = process_sp->GetABI().get();
            if (abi)
            {
                m_unwind_plan_arch_default_sp.reset (new UnwindPlan (lldb::eRegisterKindGeneric));
                if (m_unwind_plan_arch_default_sp)
                    abi->CreateDefaultUnwindPlan(*m_unwind_plan_arch_default_sp);
            }
        }
    }

    return m_unwind_plan_arch_default_sp;
}

UnwindPlanSP
FuncUnwinders::GetUnwindPlanArchitectureDefaultAtFunctionEntry (Thread& thread)
{
    Mutex::Locker locker (m_mutex);
    if (m_tried_unwind_arch_default_at_func_entry == false 
        && m_unwind_plan_arch_default_at_func_entry_sp.get() == nullptr)
    {
        m_tried_unwind_arch_default_at_func_entry = true;
        Address current_pc;
        ProcessSP process_sp (thread.CalculateProcess());
        if (process_sp)
        {
            ABI *abi = process_sp->GetABI().get();
            if (abi)
            {
                m_unwind_plan_arch_default_at_func_entry_sp.reset (new UnwindPlan (lldb::eRegisterKindGeneric));
                if (m_unwind_plan_arch_default_at_func_entry_sp)
                    abi->CreateFunctionEntryUnwindPlan(*m_unwind_plan_arch_default_at_func_entry_sp);
            }
        }
    }

    return m_unwind_plan_arch_default_at_func_entry_sp;
}


Address&
FuncUnwinders::GetFirstNonPrologueInsn (Target& target)
{
    if (m_first_non_prologue_insn.IsValid())
        return m_first_non_prologue_insn;
    ExecutionContext exe_ctx (target.shared_from_this(), false);
    UnwindAssemblySP assembly_profiler_sp (GetUnwindAssemblyProfiler());
    if (assembly_profiler_sp)
        assembly_profiler_sp->FirstNonPrologueInsn (m_range, exe_ctx, m_first_non_prologue_insn);
    return m_first_non_prologue_insn;
}

const Address&
FuncUnwinders::GetFunctionStartAddress () const
{
    return m_range.GetBaseAddress();
}

lldb::UnwindAssemblySP
FuncUnwinders::GetUnwindAssemblyProfiler ()
{
    UnwindAssemblySP assembly_profiler_sp;
    ArchSpec arch;
    if (m_unwind_table.GetArchitecture (arch))
    {
        assembly_profiler_sp = UnwindAssembly::FindPlugin (arch);
    }
    return assembly_profiler_sp;
}

Address
FuncUnwinders::GetLSDAAddress ()
{
    Address lsda_addr;
    Mutex::Locker locker (m_mutex);

    GetUnwindPlanAtCallSite (-1);

    if (m_unwind_plan_call_site_sp && m_unwind_plan_call_site_sp->GetLSDAAddress().IsValid())
    {
        lsda_addr = m_unwind_plan_call_site_sp->GetLSDAAddress().IsValid();
    }

    return lsda_addr;
}


Address
FuncUnwinders::GetPersonalityRoutinePtrAddress ()
{
    Address personality_addr;
    Mutex::Locker locker (m_mutex);

    GetUnwindPlanAtCallSite (-1);

    if (m_unwind_plan_call_site_sp && m_unwind_plan_call_site_sp->GetPersonalityFunctionPtr().IsValid())
    {
        personality_addr = m_unwind_plan_call_site_sp->GetPersonalityFunctionPtr().IsValid();
    }

    return personality_addr;
}
