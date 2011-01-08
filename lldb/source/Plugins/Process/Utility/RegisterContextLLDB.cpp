//===-- RegisterContextLLDB.cpp --------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/lldb-private.h"
#include "RegisterContextLLDB.h"
#include "lldb/Target/Thread.h"
#include "lldb/Symbol/Function.h"
#include "lldb/Symbol/SymbolContext.h"
#include "lldb/Symbol/Symbol.h"
#include "lldb/Core/Address.h"
#include "lldb/Core/AddressRange.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/Process.h"
#include "lldb/Utility/ArchDefaultUnwindPlan.h"
#include "lldb/Symbol/FuncUnwinders.h"
#include "lldb/Core/DataBufferHeap.h"
#include "lldb/Utility/ArchVolatileRegs.h"
#include "lldb/Core/Log.h"
#include "lldb/Expression/DWARFExpression.h"
#include "lldb/Core/Value.h"
#include "lldb/Target/ExecutionContext.h"
#include "lldb/Target/StackFrame.h"

using namespace lldb;
using namespace lldb_private;


RegisterContextLLDB::RegisterContextLLDB 
(
    Thread& thread, 
    const RegisterContextSP &next_frame,
    SymbolContext& sym_ctx,
    uint32_t frame_number
) :
    RegisterContext (thread, frame_number), 
    m_thread(thread), 
    m_next_frame(next_frame), 
    m_sym_ctx(sym_ctx), 
    m_all_registers_available(false), 
    m_registers(),
    m_cfa (LLDB_INVALID_ADDRESS), 
    m_start_pc (), 
    m_current_pc (), 
    m_frame_number (frame_number),
    m_full_unwind_plan(NULL), 
    m_fast_unwind_plan(NULL), 
    m_frame_type (-1), 
    m_current_offset (0), 
    m_current_offset_backed_up_one (0), 
    m_sym_ctx_valid (false)
{
    m_sym_ctx.Clear();
    m_sym_ctx_valid = false;

    if (IsFrameZero ())
    {
        InitializeZerothFrame ();
    }
    else
    {
        InitializeNonZerothFrame ();
    }

    // This same code exists over in the GetFullUnwindPlanForFrame() but it may not have been executed yet
    bool behaves_like_zeroth_frame = false;
    if (IsFrameZero())
    {
        behaves_like_zeroth_frame = true;
    }
    if (!IsFrameZero() && ((RegisterContextLLDB*) m_next_frame.get())->m_frame_type == eSigtrampFrame)
    {
        behaves_like_zeroth_frame = true;
    }
    if (!IsFrameZero() && ((RegisterContextLLDB*) m_next_frame.get())->m_frame_type == eDebuggerFrame)
    {
        behaves_like_zeroth_frame = true;
    }
    if (behaves_like_zeroth_frame)
    {
        m_all_registers_available = true;
    }
}

// Initialize a RegisterContextLLDB which is the first frame of a stack -- the zeroth frame or currently
// executing frame.

void
RegisterContextLLDB::InitializeZerothFrame()
{
    StackFrameSP frame_sp (m_thread.GetStackFrameAtIndex (0));

    if (m_thread.GetRegisterContext() == NULL)
    {
        m_frame_type = eNotAValidFrame;
        return;
    }
    m_sym_ctx = frame_sp->GetSymbolContext (eSymbolContextFunction | eSymbolContextSymbol);
    m_sym_ctx_valid = true;
    AddressRange addr_range;
    m_sym_ctx.GetAddressRange (eSymbolContextFunction | eSymbolContextSymbol, addr_range);
    
    m_current_pc = frame_sp->GetFrameCodeAddress();

    static ConstString sigtramp_name ("_sigtramp");
    if ((m_sym_ctx.function && m_sym_ctx.function->GetMangled().GetMangledName() == sigtramp_name)
        || (m_sym_ctx.symbol && m_sym_ctx.symbol->GetMangled().GetMangledName() == sigtramp_name))
    {
        m_frame_type = eSigtrampFrame;
    }
    else
    {
        // FIXME:  Detect eDebuggerFrame here.
        m_frame_type = eNormalFrame;
    }

    // If we were able to find a symbol/function, set addr_range to the bounds of that symbol/function.
    // else treat the current pc value as the start_pc and record no offset.
    if (addr_range.GetBaseAddress().IsValid())
    {
        m_start_pc = addr_range.GetBaseAddress();
        m_current_offset = frame_sp->GetFrameCodeAddress().GetOffset() - m_start_pc.GetOffset();
        m_current_offset_backed_up_one = m_current_offset;
    }
    else
    {
        m_start_pc = m_current_pc;
        m_current_offset = -1;
        m_current_offset_backed_up_one = -1;
    }

    // We've set m_frame_type and m_sym_ctx before these calls.

    m_fast_unwind_plan = GetFastUnwindPlanForFrame ();
    m_full_unwind_plan = GetFullUnwindPlanForFrame (); 

    const UnwindPlan::Row *active_row = NULL;
    int cfa_offset = 0;
    int row_register_kind;
    if (m_full_unwind_plan && m_full_unwind_plan->PlanValidAtAddress (m_current_pc))
    {
        active_row = m_full_unwind_plan->GetRowForFunctionOffset (m_current_offset);
        row_register_kind = m_full_unwind_plan->GetRegisterKind ();
    }

    if (active_row == NULL)
    {
        m_frame_type = eNotAValidFrame;
        return;
    }

    addr_t cfa_regval;
    if (!ReadGPRValue (row_register_kind, active_row->GetCFARegister(), cfa_regval))
    {
        m_frame_type = eNotAValidFrame;
        return;
    }
    else
    {
    }
    cfa_offset = active_row->GetCFAOffset ();

    m_cfa = cfa_regval + cfa_offset;

    LogSP log(GetLogIfAllCategoriesSet (LIBLLDB_LOG_UNWIND));

    // A couple of sanity checks..
    if (cfa_regval == LLDB_INVALID_ADDRESS || cfa_regval == 0 || cfa_regval == 1)
    {
        if (log)
        {   
            log->Printf("%*sFrame %u could not find a valid cfa address",
                        m_frame_number < 100 ? m_frame_number : 100, "", m_frame_number);
        }
        m_frame_type = eNotAValidFrame;
        return;
    }

    if (log)
    {
        log->Printf("%*sThread %d Frame %u initialized frame current pc is 0x%llx cfa is 0x%llx using %s UnwindPlan", 
                    m_frame_number < 100 ? m_frame_number : 100, "", m_thread.GetIndexID(), m_frame_number,
                    (uint64_t) m_current_pc.GetLoadAddress (&m_thread.GetProcess().GetTarget()), (uint64_t) m_cfa,
                    m_full_unwind_plan->GetSourceName().GetCString());
    }
}

// Initialize a RegisterContextLLDB for the non-zeroth frame -- rely on the RegisterContextLLDB "below" it
// to provide things like its current pc value.

void
RegisterContextLLDB::InitializeNonZerothFrame()
{
    LogSP log(GetLogIfAllCategoriesSet (LIBLLDB_LOG_UNWIND));
    if (IsFrameZero ())
    {
        m_frame_type = eNotAValidFrame;
        return;
    }
    if (!((RegisterContextLLDB*)m_next_frame.get())->IsValid())
    {
        m_frame_type = eNotAValidFrame;
        return;
    }
    if (m_thread.GetRegisterContext() == NULL)
    {
        m_frame_type = eNotAValidFrame;
        return;
    }

    addr_t pc;
    if (!ReadGPRValue (eRegisterKindGeneric, LLDB_REGNUM_GENERIC_PC, pc))
    {
        if (log)
        {
            log->Printf("%*sFrame %u could not get pc value",
                        m_frame_number < 100 ? m_frame_number : 100, "", m_frame_number);
        }
        m_frame_type = eNotAValidFrame;
        return;
    }
    // A pc value of 0 up on the stack indicates we've hit the end of the stack
    if (pc == 0)
    {
        m_frame_type = eNotAValidFrame;
        return;
    }
    m_thread.GetProcess().GetTarget().GetSectionLoadList().ResolveLoadAddress (pc, m_current_pc);

    // If we don't have a Module for some reason, we're not going to find symbol/function information - just
    // stick in some reasonable defaults and hope we can unwind past this frame.
    if (!m_current_pc.IsValid() || m_current_pc.GetModule() == NULL)
    {
        if (log)
        {
            log->Printf("%*sFrame %u using architectural default unwind method",
                        m_frame_number < 100 ? m_frame_number : 100, "", m_frame_number);
        }
        ArchSpec arch = m_thread.GetProcess().GetTarget().GetArchitecture ();
        ArchDefaultUnwindPlan *arch_default = ArchDefaultUnwindPlan::FindPlugin (arch);
        if (arch_default)
        {
            m_fast_unwind_plan = NULL;
            m_full_unwind_plan = arch_default->GetArchDefaultUnwindPlan (m_thread, m_current_pc);
            m_frame_type = eNormalFrame;
            m_all_registers_available = false;
            m_current_offset = -1;
            m_current_offset_backed_up_one = -1;
            addr_t cfa_regval;
            int row_register_kind = m_full_unwind_plan->GetRegisterKind ();
            uint32_t cfa_regnum = m_full_unwind_plan->GetRowForFunctionOffset(0)->GetCFARegister();
            int cfa_offset = m_full_unwind_plan->GetRowForFunctionOffset(0)->GetCFAOffset();
            if (!ReadGPRValue (row_register_kind, cfa_regnum, cfa_regval))
            {
                if (log)
                {
                    log->Printf("%*sFrame %u failed to get cfa value",
                                m_frame_number < 100 ? m_frame_number : 100, "", m_frame_number);
                }
                m_frame_type = eNormalFrame;
                return;
            }
            m_cfa = cfa_regval + cfa_offset;

            // A couple of sanity checks..
            if (cfa_regval == LLDB_INVALID_ADDRESS || cfa_regval == 0 || cfa_regval == 1)
            {
                if (log)
                {
                    log->Printf("%*sFrame %u could not find a valid cfa address",
                                m_frame_number < 100 ? m_frame_number : 100, "", m_frame_number);
                }
                m_frame_type = eNotAValidFrame;
                return;
            }

            if (log)
            {
                log->Printf("%*sFrame %u initialized frame cfa is 0x%llx",
                            m_frame_number < 100 ? m_frame_number : 100, "", m_frame_number,
                            (uint64_t) m_cfa);
            }
            return;
        }
        m_frame_type = eNotAValidFrame;
        return;
    }

    // We require that eSymbolContextSymbol be successfully filled in or this context is of no use to us.
    if ((m_current_pc.GetModule()->ResolveSymbolContextForAddress (m_current_pc, eSymbolContextFunction| eSymbolContextSymbol, m_sym_ctx) & eSymbolContextSymbol) == eSymbolContextSymbol)
    {
        m_sym_ctx_valid = true;
    }

    AddressRange addr_range;
    if (!m_sym_ctx.GetAddressRange (eSymbolContextFunction | eSymbolContextSymbol, addr_range))
    {
        m_sym_ctx_valid = false;
    }

    bool decr_pc_and_recompute_addr_range = false;

    // If the symbol lookup failed...
    if (m_sym_ctx_valid == false)
       decr_pc_and_recompute_addr_range = true;

    // Or if we're in the middle of the stack (and not "above" an asynchronous event like sigtramp),
    // and our "current" pc is the start of a function...
    if (m_sym_ctx_valid
        && ((RegisterContextLLDB*) m_next_frame.get())->m_frame_type != eSigtrampFrame
        && ((RegisterContextLLDB*) m_next_frame.get())->m_frame_type != eDebuggerFrame
        && addr_range.GetBaseAddress().IsValid()
        && addr_range.GetBaseAddress().GetSection() == m_current_pc.GetSection()
        && addr_range.GetBaseAddress().GetOffset() == m_current_pc.GetOffset())
    {
        decr_pc_and_recompute_addr_range = true;
    }

    // We need to back up the pc by 1 byte and re-search for the Symbol to handle the case where the "saved pc"
    // value is pointing to the next function, e.g. if a function ends with a CALL instruction.
    // FIXME this may need to be an architectural-dependent behavior; if so we'll need to add a member function
    // to the ABI plugin and consult that.
    if (decr_pc_and_recompute_addr_range) 
    {
        Address temporary_pc(m_current_pc);
        temporary_pc.SetOffset(m_current_pc.GetOffset() - 1);
        m_sym_ctx.Clear();
        m_sym_ctx_valid = false;
        if ((m_current_pc.GetModule()->ResolveSymbolContextForAddress (temporary_pc, eSymbolContextFunction| eSymbolContextSymbol, m_sym_ctx) & eSymbolContextSymbol) == eSymbolContextSymbol)
        {
            m_sym_ctx_valid = true;
        }
        if (!m_sym_ctx.GetAddressRange (eSymbolContextFunction | eSymbolContextSymbol, addr_range))
        {
            m_sym_ctx_valid = false;
        }
    }

    // If we were able to find a symbol/function, set addr_range_ptr to the bounds of that symbol/function.
    // else treat the current pc value as the start_pc and record no offset.
    if (addr_range.GetBaseAddress().IsValid())
    {
        m_start_pc = addr_range.GetBaseAddress();
        m_current_offset = m_current_pc.GetOffset() - m_start_pc.GetOffset();
        m_current_offset_backed_up_one = m_current_offset;
        if (decr_pc_and_recompute_addr_range && m_current_offset_backed_up_one > 0)
            m_current_offset_backed_up_one--;
    }
    else
    {
        m_start_pc = m_current_pc;
        m_current_offset = -1;
        m_current_offset_backed_up_one = -1;
    }

    static ConstString sigtramp_name ("_sigtramp");
    if ((m_sym_ctx.function && m_sym_ctx.function->GetMangled().GetMangledName() == sigtramp_name)
        || (m_sym_ctx.symbol && m_sym_ctx.symbol->GetMangled().GetMangledName() == sigtramp_name))
    {
        m_frame_type = eSigtrampFrame;
    }
    else
    {
        // FIXME:  Detect eDebuggerFrame here.
        m_frame_type = eNormalFrame;
    }

    // We've set m_frame_type and m_sym_ctx before this call.
    m_fast_unwind_plan = GetFastUnwindPlanForFrame ();

    const UnwindPlan::Row *active_row = NULL;
    int cfa_offset = 0;
    int row_register_kind;

    // Try to get by with just the fast UnwindPlan if possible - the full UnwindPlan may be expensive to get
    // (e.g. if we have to parse the entire eh_frame section of an ObjectFile for the first time.)

    if (m_fast_unwind_plan && m_fast_unwind_plan->PlanValidAtAddress (m_current_pc))
    {
        active_row = m_fast_unwind_plan->GetRowForFunctionOffset (m_current_offset);
        row_register_kind = m_fast_unwind_plan->GetRegisterKind ();
    }
    else 
    {
        m_full_unwind_plan = GetFullUnwindPlanForFrame ();
        if (m_full_unwind_plan && m_full_unwind_plan->PlanValidAtAddress (m_current_pc))
        {
            active_row = m_full_unwind_plan->GetRowForFunctionOffset (m_current_offset);
            row_register_kind = m_full_unwind_plan->GetRegisterKind ();
        }
    }

    if (active_row == NULL)
    {
        m_frame_type = eNotAValidFrame;
        return;
    }

    addr_t cfa_regval;
    if (!ReadGPRValue (row_register_kind, active_row->GetCFARegister(), cfa_regval))
    {
        if (log)
        {
            log->Printf("%*sFrame %u failed to get cfa reg %d/%d",
                        m_frame_number < 100 ? m_frame_number : 100, "", m_frame_number,
                        row_register_kind, active_row->GetCFARegister());
        }
        m_frame_type = eNotAValidFrame;
        return;
    }
    cfa_offset = active_row->GetCFAOffset ();

    m_cfa = cfa_regval + cfa_offset;

    // A couple of sanity checks..
    if (cfa_regval == LLDB_INVALID_ADDRESS || cfa_regval == 0 || cfa_regval == 1)
    { 
        if (log)
        {
            log->Printf("%*sFrame %u could not find a valid cfa address",
                        m_frame_number < 100 ? m_frame_number : 100, "", m_frame_number);
        }
        m_frame_type = eNotAValidFrame;
        return;
    }

    if (log)
    {
        log->Printf("%*sFrame %u initialized frame current pc is 0x%llx cfa is 0x%llx", 
                    m_frame_number < 100 ? m_frame_number : 100, "", m_frame_number,
                    (uint64_t) m_current_pc.GetLoadAddress (&m_thread.GetProcess().GetTarget()), (uint64_t) m_cfa);
    }
}


bool
RegisterContextLLDB::IsFrameZero () const
{
    if (m_next_frame.get () == NULL)
        return true;
    else
        return false;
}


// Find a fast unwind plan for this frame, if possible.
//
// On entry to this method, 
//
//   1. m_frame_type should already be set to eSigtrampFrame/eDebuggerFrame if either of those are correct, 
//   2. m_sym_ctx should already be filled in, and
//   3. m_current_pc should have the current pc value for this frame
//   4. m_current_offset_backed_up_one should have the current byte offset into the function, maybe backed up by 1, -1 if unknown

UnwindPlan *
RegisterContextLLDB::GetFastUnwindPlanForFrame ()
{
    if (!m_current_pc.IsValid() || m_current_pc.GetModule() == NULL || m_current_pc.GetModule()->GetObjectFile() == NULL)
    {
        return NULL;
    }

    if (IsFrameZero ())
    {
        return NULL;
    }

    FuncUnwindersSP fu;
    fu = m_current_pc.GetModule()->GetObjectFile()->GetUnwindTable().GetFuncUnwindersContainingAddress (m_current_pc, m_sym_ctx);
    if (fu.get() == NULL)
    {
        return NULL;
    }

    // If we're in _sigtramp(), unwinding past this frame requires special knowledge.  
    if (m_frame_type == eSigtrampFrame || m_frame_type == eDebuggerFrame)
    {
        return NULL;
    }

    if (fu->GetUnwindPlanFastUnwind (m_thread) 
        && fu->GetUnwindPlanFastUnwind (m_thread)->PlanValidAtAddress (m_current_pc))
    {
        LogSP log(GetLogIfAllCategoriesSet (LIBLLDB_LOG_UNWIND));
        if (log && IsLogVerbose())
        {
            const char *has_fast = "";
            if (m_fast_unwind_plan)
                has_fast = ", and has a fast UnwindPlan";
            log->Printf("%*sFrame %u frame has a fast UnwindPlan",
                        m_frame_number < 100 ? m_frame_number : 100, "", m_frame_number);
        }
        m_frame_type = eNormalFrame;
        return fu->GetUnwindPlanFastUnwind (m_thread);
    }

    return NULL;
}

// On entry to this method, 
//
//   1. m_frame_type should already be set to eSigtrampFrame/eDebuggerFrame if either of those are correct, 
//   2. m_sym_ctx should already be filled in, and
//   3. m_current_pc should have the current pc value for this frame
//   4. m_current_offset_backed_up_one should have the current byte offset into the function, maybe backed up by 1, -1 if unknown

UnwindPlan *
RegisterContextLLDB::GetFullUnwindPlanForFrame ()
{
    LogSP log(GetLogIfAllCategoriesSet (LIBLLDB_LOG_UNWIND));
    UnwindPlan *up;
    UnwindPlan *arch_default_up = NULL;
    ArchSpec arch = m_thread.GetProcess().GetTarget().GetArchitecture ();
    ArchDefaultUnwindPlan *arch_default = ArchDefaultUnwindPlan::FindPlugin (arch);
    if (arch_default)
    {
        arch_default_up = arch_default->GetArchDefaultUnwindPlan (m_thread, m_current_pc);
    }

    bool behaves_like_zeroth_frame = false;
    if (IsFrameZero ())
    {
        behaves_like_zeroth_frame = true;
    }
    if (!IsFrameZero () && ((RegisterContextLLDB*) m_next_frame.get())->m_frame_type == eSigtrampFrame)
    {
        behaves_like_zeroth_frame = true;
    }
    if (!IsFrameZero () && ((RegisterContextLLDB*) m_next_frame.get())->m_frame_type == eDebuggerFrame)
    {
        behaves_like_zeroth_frame = true;
    }

    // If this frame behaves like a 0th frame (currently executing or interrupted asynchronously), all registers
    // can be retrieved.
    if (behaves_like_zeroth_frame)
    {
        m_all_registers_available = true;
    }

    // No Module for the current pc, try using the architecture default unwind.
    if (!m_current_pc.IsValid() || m_current_pc.GetModule() == NULL || m_current_pc.GetModule()->GetObjectFile() == NULL)
    {
        m_frame_type = eNormalFrame;
        return arch_default_up;
    }

    FuncUnwindersSP fu;
    if (m_sym_ctx_valid)
    {
        fu = m_current_pc.GetModule()->GetObjectFile()->GetUnwindTable().GetFuncUnwindersContainingAddress (m_current_pc, m_sym_ctx);
    }

    // No FuncUnwinders available for this pc, try using architectural default unwind.
    if (fu.get() == NULL)
    {
        m_frame_type = eNormalFrame;
        return arch_default_up;
    }

    // If we're in _sigtramp(), unwinding past this frame requires special knowledge.  On Mac OS X this knowledge
    // is properly encoded in the eh_frame section, so prefer that if available.
    // On other platforms we may need to provide a platform-specific UnwindPlan which encodes the details of
    // how to unwind out of sigtramp.
    if (m_frame_type == eSigtrampFrame)
    {
        m_fast_unwind_plan = NULL;
        up = fu->GetUnwindPlanAtCallSite (m_current_offset_backed_up_one);
        if (up && up->PlanValidAtAddress (m_current_pc))
        {
            return up;
        }
    }

    
    // Typically the NonCallSite UnwindPlan is the unwind created by inspecting the assembly language instructions
    up = fu->GetUnwindPlanAtNonCallSite (m_thread);
    if (behaves_like_zeroth_frame && up && up->PlanValidAtAddress (m_current_pc))
    {
        if (log && IsLogVerbose())
        {
            log->Printf("%*sFrame %u frame uses %s for full UnwindPlan",
                        m_frame_number < 100 ? m_frame_number : 100, "", m_frame_number,
                        up->GetSourceName().GetCString());
        }
        return up;
    }

    // Typically this is unwind info from an eh_frame section intended for exception handling; only valid at call sites
    up = fu->GetUnwindPlanAtCallSite (m_current_offset_backed_up_one);
    if (up && up->PlanValidAtAddress (m_current_pc))
    {
        if (log && IsLogVerbose())
        {
            log->Printf("%*sFrame %u frame uses %s for full UnwindPlan",
                        m_frame_number < 100 ? m_frame_number : 100, "", m_frame_number,
                        up->GetSourceName().GetCString());
        }
        return up;
    }
    
    // We'd prefer to use an UnwindPlan intended for call sites when we're at a call site but if we've
    // struck out on that, fall back to using the non-call-site assembly inspection UnwindPlan if possible.
    up = fu->GetUnwindPlanAtNonCallSite (m_thread);
    if (up && up->PlanValidAtAddress (m_current_pc))
    {
        if (log && IsLogVerbose())
        {
            log->Printf("%*sFrame %u frame uses %s for full UnwindPlan",
                        m_frame_number < 100 ? m_frame_number : 100, "", m_frame_number,
                        up->GetSourceName().GetCString());
        }
        return up;
    }

    // If nothing else, use the architectural default UnwindPlan and hope that does the job.
    if (log && IsLogVerbose())
    {
        log->Printf("%*sFrame %u frame uses %s for full UnwindPlan",
                    m_frame_number < 100 ? m_frame_number : 100, "", m_frame_number,
                    arch_default_up->GetSourceName().GetCString());
    }
    return arch_default_up;
}


void
RegisterContextLLDB::Invalidate ()
{
    m_frame_type = eNotAValidFrame;
}

size_t
RegisterContextLLDB::GetRegisterCount ()
{
    return m_thread.GetRegisterContext()->GetRegisterCount();
}

const RegisterInfo *
RegisterContextLLDB::GetRegisterInfoAtIndex (uint32_t reg)
{
    return m_thread.GetRegisterContext()->GetRegisterInfoAtIndex (reg);
}

size_t
RegisterContextLLDB::GetRegisterSetCount ()
{
    return m_thread.GetRegisterContext()->GetRegisterSetCount ();
}

const RegisterSet *
RegisterContextLLDB::GetRegisterSet (uint32_t reg_set)
{
    return m_thread.GetRegisterContext()->GetRegisterSet (reg_set);
}

uint32_t
RegisterContextLLDB::ConvertRegisterKindToRegisterNumber (uint32_t kind, uint32_t num)
{
    return m_thread.GetRegisterContext()->ConvertRegisterKindToRegisterNumber (kind, num);
}

bool
RegisterContextLLDB::ReadRegisterBytesFromRegisterLocation (uint32_t regnum, RegisterLocation regloc, DataExtractor &data)
{
    if (!IsValid())
        return false;

    if (regloc.type == eRegisterInRegister)
    {
        data.SetAddressByteSize (m_thread.GetProcess().GetAddressByteSize());
        data.SetByteOrder (m_thread.GetProcess().GetByteOrder());
        if (IsFrameZero ())
        {
            return m_thread.GetRegisterContext()->ReadRegisterBytes (regloc.location.register_number, data);
        }
        else
        {
            return m_next_frame->ReadRegisterBytes (regloc.location.register_number, data);
        }
    }
    if (regloc.type == eRegisterNotSaved)
    {
        return false;
    }
    if (regloc.type == eRegisterSavedAtHostMemoryLocation)
    {
        assert ("FIXME debugger inferior function call unwind");
    }
    if (regloc.type != eRegisterSavedAtMemoryLocation)
    {
        assert ("Unknown RegisterLocation type.");
    }

    const RegisterInfo *reg_info = m_thread.GetRegisterContext()->GetRegisterInfoAtIndex (regnum);
    DataBufferSP data_sp (new DataBufferHeap (reg_info->byte_size, 0));
    data.SetData (data_sp, 0, reg_info->byte_size);
    data.SetAddressByteSize (m_thread.GetProcess().GetAddressByteSize());

    if (regloc.type == eRegisterValueInferred)
    {
        data.SetByteOrder (eByteOrderHost);
        switch (reg_info->byte_size)
        {
            case 1:
            {
                uint8_t val = regloc.location.register_value;
                memcpy (data_sp->GetBytes(), &val, sizeof (val));
                data.SetByteOrder (eByteOrderHost);
                return true;
            }
            case 2:
            {
                uint16_t val = regloc.location.register_value;
                memcpy (data_sp->GetBytes(), &val, sizeof (val));
                data.SetByteOrder (eByteOrderHost);
                return true;
            }
            case 4:
            {
                uint32_t val = regloc.location.register_value;
                memcpy (data_sp->GetBytes(), &val, sizeof (val));
                data.SetByteOrder (eByteOrderHost);
                return true;
            }
            case 8:
            {
                uint64_t val = regloc.location.register_value;
                memcpy (data_sp->GetBytes(), &val, sizeof (val));
                data.SetByteOrder (eByteOrderHost);
                return true;
            }
        }
        return false;
    }

    assert (regloc.type == eRegisterSavedAtMemoryLocation);
    Error error;
    data.SetByteOrder (m_thread.GetProcess().GetByteOrder());
    if (!m_thread.GetProcess().ReadMemory (regloc.location.target_memory_location, data_sp->GetBytes(), reg_info->byte_size, error))
        return false;
    return true;
}

bool
RegisterContextLLDB::WriteRegisterBytesToRegisterLocation (uint32_t regnum, RegisterLocation regloc, DataExtractor &data, uint32_t data_offset)
{
    if (!IsValid())
        return false;

    if (regloc.type == eRegisterInRegister)
    {
        if (IsFrameZero ())
        {
            return m_thread.GetRegisterContext()->WriteRegisterBytes (regloc.location.register_number, data, data_offset);
        }
        else
        {
            return m_next_frame->WriteRegisterBytes (regloc.location.register_number, data, data_offset);
        }
    }
    if (regloc.type == eRegisterNotSaved)
    {
        return false;
    }
    if (regloc.type == eRegisterValueInferred)
    {
        return false;
    }
    if (regloc.type == eRegisterSavedAtHostMemoryLocation)
    {
        assert ("FIXME debugger inferior function call unwind");
    }
    if (regloc.type != eRegisterSavedAtMemoryLocation)
    {
        assert ("Unknown RegisterLocation type.");
    }

    Error error;
    const RegisterInfo *reg_info = m_thread.GetRegisterContext()->GetRegisterInfoAtIndex (regnum);
    if (reg_info->byte_size == 0)
        return false;
    uint8_t *buf = (uint8_t*) alloca (reg_info->byte_size);
    if (data.ExtractBytes (data_offset, reg_info->byte_size, m_thread.GetProcess().GetByteOrder(), buf) != reg_info->byte_size)
        return false;
    if (m_thread.GetProcess().WriteMemory (regloc.location.target_memory_location, buf, reg_info->byte_size, error) != reg_info->byte_size)
        return false;

    return true;
}


bool
RegisterContextLLDB::IsValid () const
{
    return m_frame_type != eNotAValidFrame;
}

// Answer the question: Where did THIS frame save the CALLER frame ("previous" frame)'s register value?

bool
RegisterContextLLDB::SavedLocationForRegister (uint32_t lldb_regnum, RegisterLocation &regloc)
{
    LogSP log(GetLogIfAllCategoriesSet (LIBLLDB_LOG_UNWIND));

    // Have we already found this register location?
    std::map<uint32_t, RegisterLocation>::const_iterator iterator;
    if (m_registers.size() > 0)
    {
        iterator = m_registers.find (lldb_regnum);
        if (iterator != m_registers.end())
        {
            regloc = iterator->second;
            return true;
        }
    }

    // Are we looking for the CALLER's stack pointer?  The stack pointer is defined to be the same as THIS frame's
    // CFA so just return the CFA value.  This is true on x86-32/x86-64 at least.
    uint32_t sp_regnum;
    if (m_thread.GetRegisterContext()->ConvertBetweenRegisterKinds (eRegisterKindGeneric, LLDB_REGNUM_GENERIC_SP, eRegisterKindLLDB, sp_regnum)
        && sp_regnum == lldb_regnum)
    {
        // make sure we won't lose precision copying an addr_t (m_cfa) into a uint64_t (.register_value)
        assert (sizeof (addr_t) <= sizeof (uint64_t));
        regloc.type = eRegisterValueInferred;
        regloc.location.register_value = m_cfa;
        m_registers[lldb_regnum] = regloc;
        return true;
    }

    // Look through the available UnwindPlans for the register location.

    UnwindPlan::Row::RegisterLocation unwindplan_regloc;
    bool have_unwindplan_regloc = false;
    int unwindplan_registerkind = -1;

    if (m_fast_unwind_plan)
    {
        const UnwindPlan::Row *active_row = m_fast_unwind_plan->GetRowForFunctionOffset (m_current_offset);
        unwindplan_registerkind = m_fast_unwind_plan->GetRegisterKind ();
        uint32_t row_regnum;
        if (!m_thread.GetRegisterContext()->ConvertBetweenRegisterKinds (eRegisterKindLLDB, lldb_regnum, unwindplan_registerkind, row_regnum))
        {
            if (log)
            {
                log->Printf("%*sFrame %u could not convert lldb regnum %d into %d RegisterKind reg numbering scheme",
                            m_frame_number < 100 ? m_frame_number : 100, "", m_frame_number,
                            lldb_regnum, (int) unwindplan_registerkind);
            }
            return false;
        }
        if (active_row->GetRegisterInfo (row_regnum, unwindplan_regloc))
        {
            if (log)
            {
                log->Printf("%*sFrame %u supplying caller's saved reg %d's location using FastUnwindPlan",
                            m_frame_number < 100 ? m_frame_number : 100, "", m_frame_number,
                            lldb_regnum);
            }
            have_unwindplan_regloc = true;
        }
    }

    if (!have_unwindplan_regloc)
    {
        // m_full_unwind_plan being NULL means that we haven't tried to find a full UnwindPlan yet
        if (m_full_unwind_plan == NULL)
        {
            m_full_unwind_plan = GetFullUnwindPlanForFrame ();
        }
        if (m_full_unwind_plan)
        {
            const UnwindPlan::Row *active_row = m_full_unwind_plan->GetRowForFunctionOffset (m_current_offset);
            unwindplan_registerkind = m_full_unwind_plan->GetRegisterKind ();
            uint32_t row_regnum;
            if (!m_thread.GetRegisterContext()->ConvertBetweenRegisterKinds (eRegisterKindLLDB, lldb_regnum, unwindplan_registerkind, row_regnum))
            {
                if (log)
                {
                    if (unwindplan_registerkind == eRegisterKindGeneric)
                        log->Printf("%*sFrame %u could not convert lldb regnum %d into eRegisterKindGeneric reg numbering scheme",
                                    m_frame_number < 100 ? m_frame_number : 100, "", m_frame_number,
                                    lldb_regnum);
                    else
                        log->Printf("%*sFrame %u could not convert lldb regnum %d into %d RegisterKind reg numbering scheme",
                                    m_frame_number < 100 ? m_frame_number : 100, "", m_frame_number,
                                    lldb_regnum, (int) unwindplan_registerkind);
                }
                return false;
            }

            if (active_row->GetRegisterInfo (row_regnum, unwindplan_regloc))
            {
                have_unwindplan_regloc = true;
                if (log && IsLogVerbose ())
                {                
                    log->Printf("%*sFrame %u supplying caller's saved reg %d's location using %s UnwindPlan",
                                m_frame_number < 100 ? m_frame_number : 100, "", m_frame_number,
                                lldb_regnum, m_full_unwind_plan->GetSourceName().GetCString());
                }
            }
        }
    }

    if (have_unwindplan_regloc == false)
    {
        // If a volatile register is being requested, we don't want to forward m_next_frame's register contents 
        // up the stack -- the register is not retrievable at this frame.
        ArchSpec arch = m_thread.GetProcess().GetTarget().GetArchitecture ();
        ArchVolatileRegs *volatile_regs = ArchVolatileRegs::FindPlugin (arch);
        if (volatile_regs && volatile_regs->RegisterIsVolatile (m_thread, lldb_regnum))
        {
            if (log)
            {
                log->Printf("%*sFrame %u did not supply reg location for %d because it is volatile",
                            m_frame_number < 100 ? m_frame_number : 100, "", m_frame_number,
                            lldb_regnum);
            }
            return false;
        }  

        if (!IsFrameZero ())
        {
            return ((RegisterContextLLDB*)m_next_frame.get())->SavedLocationForRegister (lldb_regnum, regloc);
        }
        else
        {
            // This is frame 0 - we should return the actual live register context value
            RegisterLocation new_regloc;
            new_regloc.type = eRegisterInRegister;
            new_regloc.location.register_number = lldb_regnum;
            m_registers[lldb_regnum] = new_regloc;
            regloc = new_regloc;
            return true;
        }
        if (log)
        {
            log->Printf("%*sFrame %u could not supply caller's reg %d location",
                        m_frame_number < 100 ? m_frame_number : 100, "", m_frame_number,
                        lldb_regnum);
        }
        return false;
    }

    // unwindplan_regloc has valid contents about where to retrieve the register
    if (unwindplan_regloc.IsUnspecified())
    {
        RegisterLocation new_regloc;
        new_regloc.type = eRegisterNotSaved;
        m_registers[lldb_regnum] = new_regloc;
        if (log)
        {
            log->Printf("%*sFrame %u could not supply caller's reg %d location",
                        m_frame_number < 100 ? m_frame_number : 100, "", m_frame_number,
                        lldb_regnum);
        }
        return false;
    }

    if (unwindplan_regloc.IsSame())
    {
        if (!IsFrameZero ())
        {
            return ((RegisterContextLLDB*)m_next_frame.get())->SavedLocationForRegister (lldb_regnum, regloc);
        }
        else
        {
            if (log)
            {
                log->Printf("%*sFrame %u could not supply caller's reg %d location",
                            m_frame_number < 100 ? m_frame_number : 100, "", m_frame_number,
                            lldb_regnum);
            }
            return false;
        }
    }

    if (unwindplan_regloc.IsCFAPlusOffset())
    {
        int offset = unwindplan_regloc.GetOffset();
        regloc.type = eRegisterValueInferred;
        regloc.location.register_value = m_cfa + offset;
        m_registers[lldb_regnum] = regloc;
        return true;
    }

    if (unwindplan_regloc.IsAtCFAPlusOffset())
    {
        int offset = unwindplan_regloc.GetOffset();
        regloc.type = eRegisterSavedAtMemoryLocation;
        regloc.location.target_memory_location = m_cfa + offset;
        m_registers[lldb_regnum] = regloc;
        return true;
    }

    if (unwindplan_regloc.IsInOtherRegister())
    {
        uint32_t unwindplan_regnum = unwindplan_regloc.GetRegisterNumber();
        uint32_t row_regnum_in_lldb;
        if (!m_thread.GetRegisterContext()->ConvertBetweenRegisterKinds (unwindplan_registerkind, unwindplan_regnum, eRegisterKindLLDB, row_regnum_in_lldb))
        {
            if (log)
            {
                log->Printf("%*sFrame %u could not supply caller's reg %d location",
                            m_frame_number < 100 ? m_frame_number : 100, "", m_frame_number,
                            lldb_regnum);
            }
            return false;
        }
        regloc.type = eRegisterInRegister;
        regloc.location.register_number = row_regnum_in_lldb;
        m_registers[lldb_regnum] = regloc;
        return true;
    }

    if (unwindplan_regloc.IsDWARFExpression() || unwindplan_regloc.IsAtDWARFExpression())
    {
        DataExtractor dwarfdata (unwindplan_regloc.GetDWARFExpressionBytes(), 
                                 unwindplan_regloc.GetDWARFExpressionLength(), 
                                 m_thread.GetProcess().GetByteOrder(), m_thread.GetProcess().GetAddressByteSize());
        DWARFExpression dwarfexpr (dwarfdata, 0, unwindplan_regloc.GetDWARFExpressionLength());
        dwarfexpr.SetRegisterKind (unwindplan_registerkind);
        ExecutionContext exe_ctx (&m_thread.GetProcess(), &m_thread, NULL);
        Value result;
        Error error;
        if (dwarfexpr.Evaluate (&exe_ctx, NULL, this, 0, NULL, result, &error))
        {
            addr_t val;
            val = result.GetScalar().ULongLong();
            if (unwindplan_regloc.IsDWARFExpression())
             {
                regloc.type = eRegisterValueInferred;
                regloc.location.register_value = val;
                m_registers[lldb_regnum] = regloc;
                return true;
            }
            else
            {
               regloc.type = eRegisterSavedAtMemoryLocation;
               regloc.location.target_memory_location = val;
               m_registers[lldb_regnum] = regloc;
               return true;
            }
        }
        if (log)
        {
            log->Printf("%*sFrame %u tried to use IsDWARFExpression or IsAtDWARFExpression for reg %d but failed",
                        m_frame_number < 100 ? m_frame_number : 100, "", m_frame_number,
                        lldb_regnum);
        }
        return false;
    }

    if (log)
    {
        log->Printf("%*sFrame %u could not supply caller's reg %d location",
                    m_frame_number < 100 ? m_frame_number : 100, "", m_frame_number,
                    lldb_regnum);
    }

    
    // assert ("UnwindPlan::Row types atDWARFExpression and isDWARFExpression are unsupported.");
    return false;
}

// Retrieve a general purpose register value for THIS from, as saved by the NEXT frame, i.e. the frame that
// this frame called.  e.g.
//
//  foo () { }
//  bar () { foo (); }
//  main () { bar (); }
//
//  stopped in foo() so
//     frame 0 - foo
//     frame 1 - bar
//     frame 2 - main
//  and this RegisterContext is for frame 1 (bar) - if we want to get the pc value for frame 1, we need to ask
//  where frame 0 (the "next" frame) saved that and retrieve the value.

bool
RegisterContextLLDB::ReadGPRValue (int register_kind, uint32_t regnum, addr_t &value)
{
    if (!IsValid())
        return false;

    uint32_t lldb_regnum;
    if (register_kind == eRegisterKindLLDB)
    {
        lldb_regnum = regnum;
    }
    else if (!m_thread.GetRegisterContext()->ConvertBetweenRegisterKinds (register_kind, regnum, eRegisterKindLLDB, lldb_regnum))
    {
        return false;
    }

    uint32_t offset = 0;
    DataExtractor data;
    data.SetAddressByteSize (m_thread.GetProcess().GetAddressByteSize());
    data.SetByteOrder (m_thread.GetProcess().GetByteOrder());

    // if this is frame 0 (currently executing frame), get the requested reg contents from the actual thread registers
    if (IsFrameZero ())
    {
        if (m_thread.GetRegisterContext()->ReadRegisterBytes (lldb_regnum, data))
        {
            data.SetAddressByteSize (m_thread.GetProcess().GetAddressByteSize());
            value = data.GetAddress (&offset);
            return true;
        }
        else
        {
            return false;
        }
    }

    RegisterLocation regloc;
    if (!((RegisterContextLLDB*)m_next_frame.get())->SavedLocationForRegister (lldb_regnum, regloc))
    {
        return false;
    }
    if (!ReadRegisterBytesFromRegisterLocation (lldb_regnum, regloc, data))
    {
        return false;
    }
    data.SetAddressByteSize (m_thread.GetProcess().GetAddressByteSize());
    value = data.GetAddress (&offset);
    return true;
}

// Find the value of a register in THIS frame

bool
RegisterContextLLDB::ReadRegisterBytes (uint32_t lldb_reg, DataExtractor& data)
{
    LogSP log(GetLogIfAllCategoriesSet (LIBLLDB_LOG_UNWIND));
    if (!IsValid())
        return false;

    if (log && IsLogVerbose ())
    {
        log->Printf("%*sFrame %u looking for register saved location for reg %d",
                    m_frame_number < 100 ? m_frame_number : 100, "", m_frame_number,
                    lldb_reg);
    }

    // If this is the 0th frame, hand this over to the live register context
    if (IsFrameZero ())
    {
        if (log)
        {
            log->Printf("%*sFrame %u passing along to the live register context for reg %d",
                        m_frame_number < 100 ? m_frame_number : 100, "", m_frame_number,
                        lldb_reg);
        }
        return m_thread.GetRegisterContext()->ReadRegisterBytes (lldb_reg, data);
    }

    RegisterLocation regloc;
    // Find out where the NEXT frame saved THIS frame's register contents
    if (!((RegisterContextLLDB*)m_next_frame.get())->SavedLocationForRegister (lldb_reg, regloc))
        return false;

    return ReadRegisterBytesFromRegisterLocation (lldb_reg, regloc, data);
}

bool
RegisterContextLLDB::WriteRegisterBytes (uint32_t lldb_reg, DataExtractor &data, uint32_t data_offset)
{
    LogSP log(GetLogIfAllCategoriesSet (LIBLLDB_LOG_UNWIND));
    if (!IsValid())
        return false;

    if (log && IsLogVerbose ())
    {
        log->Printf("%*sFrame %u looking for register saved location for reg %d",
                    m_frame_number < 100 ? m_frame_number : 100, "", m_frame_number,
                    lldb_reg);
    }

    // If this is the 0th frame, hand this over to the live register context
    if (IsFrameZero ())
    {
        if (log)
        {
            log->Printf("%*sFrame %u passing along to the live register context for reg %d",
                        m_frame_number < 100 ? m_frame_number : 100, "", m_frame_number,
                        lldb_reg);
        }
        return m_thread.GetRegisterContext()->WriteRegisterBytes (lldb_reg, data, data_offset);
    }

    RegisterLocation regloc;
    // Find out where the NEXT frame saved THIS frame's register contents
    if (!((RegisterContextLLDB*)m_next_frame.get())->SavedLocationForRegister (lldb_reg, regloc))
        return false;

    return WriteRegisterBytesToRegisterLocation (lldb_reg, regloc, data, data_offset);
}

// Don't need to implement this one
bool
RegisterContextLLDB::ReadAllRegisterValues (lldb::DataBufferSP &data_sp)
{
    return false;
}

// Don't need to implement this one
bool
RegisterContextLLDB::WriteAllRegisterValues (const lldb::DataBufferSP& data_sp)
{
    return false;
}

// Retrieve the pc value for THIS from

bool
RegisterContextLLDB::GetCFA (addr_t& cfa)
{
    if (!IsValid())
    {
        return false;
    }
    if (m_cfa == LLDB_INVALID_ADDRESS)
    {
        return false;
    }
    cfa = m_cfa;
    return true;
}

// Retrieve the address of the start of the function of THIS frame

bool
RegisterContextLLDB::GetStartPC (addr_t& start_pc)
{
    if (!IsValid())
        return false;
    if (!m_start_pc.IsValid())
    {
        return GetPC (start_pc); 
    }
    start_pc = m_start_pc.GetLoadAddress (&m_thread.GetProcess().GetTarget());
    return true;
}

// Retrieve the current pc value for THIS frame, as saved by the NEXT frame.

bool
RegisterContextLLDB::GetPC (addr_t& pc)
{
    if (!IsValid())
        return false;
    if (ReadGPRValue (eRegisterKindGeneric, LLDB_REGNUM_GENERIC_PC, pc))
    {
        // A pc value of 0 or 1 is impossible in the middle of the stack -- it indicates the end of a stack walk.
        // On the currently executing frame (or such a frame interrupted asynchronously by sigtramp et al) this may
        // occur if code has jumped through a NULL pointer -- we want to be able to unwind past that frame to help
        // find the bug.

        if (m_all_registers_available == false 
            && (pc == 0 || pc == 1))
        {
            return false;
        }
        else 
        {
            return true;
        }
    }
    else
    {
        return false;
    }
}
