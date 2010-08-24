//===-- StackFrame.cpp ------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Target/StackFrame.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Core/Module.h"
#include "lldb/Core/Disassembler.h"
#include "lldb/Core/Value.h"
#include "lldb/Symbol/Function.h"
#include "lldb/Target/ExecutionContext.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/RegisterContext.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/Thread.h"

using namespace lldb;
using namespace lldb_private;

// The first bits in the flags are reserved for the SymbolContext::Scope bits
// so we know if we have tried to look up information in our internal symbol
// context (m_sc) already.
#define RESOLVED_PC_SO_ADDR (uint32_t(eSymbolContextEverything + 1))
#define RESOLVED_FRAME_ID   (RESOLVED_PC_SO_ADDR << 1)
#define GOT_FRAME_BASE      (RESOLVED_FRAME_ID << 1)
#define FRAME_IS_OBSOLETE   (GOT_FRAME_BASE << 1)
#define RESOLVED_VARIABLES  (FRAME_IS_OBSOLETE << 1)

StackFrame::StackFrame 
(
    lldb::user_id_t frame_idx, 
    lldb::user_id_t concrete_frame_index, 
    Thread &thread, 
    lldb::addr_t cfa, 
    uint32_t inline_height, 
    lldb::addr_t pc, 
    const SymbolContext *sc_ptr
) :
    m_frame_index (frame_idx),
    m_concrete_frame_index (concrete_frame_index),    
    m_thread (thread),
    m_reg_context_sp (),
    m_id (cfa, inline_height),
    m_pc (NULL, pc),
    m_sc (),
    m_flags (),
    m_frame_base (),
    m_frame_base_error (),
    m_variable_list_sp (),
    m_value_object_list ()
{
    if (sc_ptr != NULL)
    {
        m_sc = *sc_ptr;
        m_flags.Set(m_sc.GetResolvedMask ());
    }
}

StackFrame::StackFrame 
(
    lldb::user_id_t frame_idx, 
    lldb::user_id_t concrete_frame_index, 
    Thread &thread, 
    const RegisterContextSP &reg_context_sp, 
    lldb::addr_t cfa, 
    uint32_t inline_height, 
    lldb::addr_t pc, 
    const SymbolContext *sc_ptr
) :
    m_frame_index (frame_idx),
    m_concrete_frame_index (concrete_frame_index),    
    m_thread (thread),
    m_reg_context_sp (reg_context_sp),
    m_id (cfa, inline_height),
    m_pc (NULL, pc),
    m_sc (),
    m_flags (),
    m_frame_base (),
    m_frame_base_error (),
    m_variable_list_sp (),
    m_value_object_list ()
{
    if (sc_ptr != NULL)
    {
        m_sc = *sc_ptr;
        m_flags.Set(m_sc.GetResolvedMask ());
    }
    
    if (reg_context_sp && !m_sc.target_sp)
    {
        m_sc.target_sp = reg_context_sp->GetThread().GetProcess().GetTarget().GetSP();
        m_flags.Set (eSymbolContextTarget);
    }
}

StackFrame::StackFrame 
(
    lldb::user_id_t frame_idx, 
    lldb::user_id_t concrete_frame_index, 
    Thread &thread, 
    const RegisterContextSP &reg_context_sp, 
    lldb::addr_t cfa, 
    uint32_t inline_height, 
    const Address& pc_addr,
    const SymbolContext *sc_ptr
) :
    m_frame_index (frame_idx),
    m_concrete_frame_index (concrete_frame_index),    
    m_thread (thread),
    m_reg_context_sp (reg_context_sp),
    m_id (cfa, inline_height),
    m_pc (pc_addr),
    m_sc (),
    m_flags (),
    m_frame_base (),
    m_frame_base_error (),
    m_variable_list_sp (),
    m_value_object_list ()
{
    if (sc_ptr != NULL)
    {
        m_sc = *sc_ptr;
        m_flags.Set(m_sc.GetResolvedMask ());
    }
    
    if (m_sc.target_sp.get() == NULL && reg_context_sp)
    {
        m_sc.target_sp = reg_context_sp->GetThread().GetProcess().GetTarget().GetSP();
        m_flags.Set (eSymbolContextTarget);
    }
    
    if (m_sc.module_sp.get() == NULL && pc_addr.GetSection())
    {
        Module *pc_module = pc_addr.GetSection()->GetModule();
        if (pc_module)
        {
            m_sc.module_sp = pc_module->GetSP();
            m_flags.Set (eSymbolContextModule);
        }
    }
}


//----------------------------------------------------------------------
// Destructor
//----------------------------------------------------------------------
StackFrame::~StackFrame()
{
}

StackID&
StackFrame::GetStackID()
{
    // Make sure we have resolved our stack ID's address range before we give
    // it out to any external clients
    if (m_id.GetStartAddress().IsValid() == 0 && m_flags.IsClear(RESOLVED_FRAME_ID))
    {
        m_flags.Set (RESOLVED_FRAME_ID);

        // Resolve our PC to section offset if we haven't alreday done so
        // and if we don't have a module. The resolved address section will
        // contain the module to which it belongs.
        if (!m_sc.module_sp && m_flags.IsClear(RESOLVED_PC_SO_ADDR))
            GetPC();

        const uint32_t resolve_scope = eSymbolContextModule |
                                       eSymbolContextCompUnit |
                                       eSymbolContextFunction;

        if (m_sc.module_sp)
        {
            if (m_sc.module_sp->ResolveSymbolContextForAddress (GetPC(), resolve_scope, m_sc) & eSymbolContextFunction)
            {
                assert (m_sc.function);
                m_id.SetStartAddress(m_sc.function->GetAddressRange().GetBaseAddress());
            }
            else if (m_sc.module_sp->ResolveSymbolContextForAddress (GetPC(), resolve_scope, m_sc) & eSymbolContextSymbol)
            {
                assert (m_sc.symbol);
                AddressRange *symbol_range_ptr = m_sc.symbol->GetAddressRangePtr();
                if (symbol_range_ptr)
                    m_id.SetStartAddress(symbol_range_ptr->GetBaseAddress());
            }
        }
//      else if (m_sc.target != NULL)
//      {
//          if (m_sc.target->GetImages().ResolveSymbolContextForAddress (GetPC(), resolve_scope, m_sc) & eSymbolContextFunction)
//          {
//              assert (m_sc.function);
//              m_id.GetAddressRange() = m_sc.function->GetAddressRange();
//          }
//          else if (m_sc.target->GetImages().ResolveSymbolContextForAddress (GetPC(), resolve_scope, m_sc) & eSymbolContextSymbol)
//          {
//              assert (m_sc.symbol);
//              AddressRange *symbol_range_ptr = m_sc.symbol->GetAddressRange();
//              if (symbol_range_ptr)
//                  m_id.GetAddressRange() = *symbol_range_ptr;
//          }
//      }
    }
    return m_id;
}

Address&
StackFrame::GetPC()
{
    if (m_flags.IsClear(RESOLVED_PC_SO_ADDR) && !m_pc.IsSectionOffset())
    {
        m_flags.Set (RESOLVED_PC_SO_ADDR);

        // Resolve the PC into a temporary address because if ResolveLoadAddress
        // fails to resolve the address, it will clear the address object...
        Address resolved_pc;
        if (m_thread.GetProcess().ResolveLoadAddress(m_pc.GetOffset(), resolved_pc))
        {
            m_pc = resolved_pc;
            const Section *section = m_pc.GetSection();
            if (section)
            {
                Module *module = section->GetModule();
                if (module)
                {
                    m_sc.module_sp = module->GetSP();
                    if (m_sc.module_sp)
                        m_flags.Set(eSymbolContextModule);
                }
            }
        }
    }
    return m_pc;
}

void
StackFrame::ChangePC (addr_t pc)
{
    m_pc.SetOffset(pc);
    m_pc.SetSection(NULL);
    m_sc.Clear();
    m_flags.SetAllFlagBits(0);
    m_thread.ClearStackFrames ();
}

const char *
StackFrame::Disassemble ()
{
    if (m_disassembly.GetSize() == 0)
    {
        ExecutionContext exe_ctx;
        Calculate(exe_ctx);
        Target &target = m_thread.GetProcess().GetTarget();
        Disassembler::Disassemble (target.GetDebugger(),
                                   target.GetArchitecture(),
                                   exe_ctx,
                                   0,
                                   false,
                                   m_disassembly);
        if (m_disassembly.GetSize() == 0)
            return NULL;
    }
    return m_disassembly.GetData();
}

//----------------------------------------------------------------------
// Get the symbol context if we already haven't done so by resolving the
// PC address as much as possible. This way when we pass around a
// StackFrame object, everyone will have as much information as
// possible and no one will ever have to look things up manually.
//----------------------------------------------------------------------
const SymbolContext&
StackFrame::GetSymbolContext (uint32_t resolve_scope)
{
    // Copy our internal symbol context into "sc".

    if ((m_flags.GetAllFlagBits() & resolve_scope) != resolve_scope)
    {
        // Resolve our PC to section offset if we haven't alreday done so
        // and if we don't have a module. The resolved address section will
        // contain the module to which it belongs
        if (!m_sc.module_sp && m_flags.IsClear(RESOLVED_PC_SO_ADDR))
            GetPC();

        // If this is not frame zero, then we need to subtract 1 from the PC
        // value when doing address lookups since the PC will be on the 
        // instruction following the function call instruction...
        
        Address lookup_addr(GetPC());
        if (m_frame_index > 0 && lookup_addr.IsValid())
        {
            addr_t offset = lookup_addr.GetOffset();
            if (offset > 0)
                lookup_addr.SetOffset(offset - 1);
        }

        if (m_sc.module_sp)
        {
            // We have something in our stack frame symbol context, lets check
            // if we haven't already tried to lookup one of those things. If we
            // haven't then we will do the query.
            
            uint32_t actual_resolve_scope = 0;
            
            if (resolve_scope & eSymbolContextCompUnit)
            {
                if (m_flags.IsClear (eSymbolContextCompUnit))
                {
                    if (m_sc.comp_unit)
                        m_flags.Set (eSymbolContextCompUnit);
                    else
                        actual_resolve_scope |= eSymbolContextCompUnit;
                }
            }

            if (resolve_scope & eSymbolContextFunction)
            {
                if (m_flags.IsClear (eSymbolContextFunction))
                {
                    if (m_sc.function)
                        m_flags.Set (eSymbolContextFunction);
                    else
                        actual_resolve_scope |= eSymbolContextFunction;
                }
            }

            if (resolve_scope & eSymbolContextBlock)
            {
                if (m_flags.IsClear (eSymbolContextBlock))
                {
                    if (m_sc.block)
                        m_flags.Set (eSymbolContextBlock);
                    else
                        actual_resolve_scope |= eSymbolContextBlock;
                }
            }

            if (resolve_scope & eSymbolContextSymbol)
            {
                if (m_flags.IsClear (eSymbolContextSymbol))
                {
                    if (m_sc.symbol)
                        m_flags.Set (eSymbolContextSymbol);
                    else
                        actual_resolve_scope |= eSymbolContextSymbol;
                }
            }

            if (resolve_scope & eSymbolContextLineEntry)
            {
                if (m_flags.IsClear (eSymbolContextLineEntry))
                {
                    if (m_sc.line_entry.IsValid())
                        m_flags.Set (eSymbolContextLineEntry);
                    else
                        actual_resolve_scope |= eSymbolContextLineEntry;
                }
            }
            
            if (actual_resolve_scope)
            {
                // We might be resolving less information than what is already
                // in our current symbol context so resolve into a temporary 
                // symbol context "sc" so we don't clear out data we have 
                // already found in "m_sc"
                SymbolContext sc;
                // Set flags that indicate what we have tried to resolve
                const uint32_t resolved = m_sc.module_sp->ResolveSymbolContextForAddress (lookup_addr, actual_resolve_scope, sc);
                // Only replace what we didn't already have as we may have 
                // information for an inlined function scope that won't match
                // what a standard lookup by address would match
                if (resolved & eSymbolContextCompUnit)  m_sc.comp_unit  = sc.comp_unit;
                if (resolved & eSymbolContextFunction)  m_sc.function   = sc.function;
                if (resolved & eSymbolContextBlock)     m_sc.block      = sc.block;
                if (resolved & eSymbolContextSymbol)    m_sc.symbol     = sc.symbol;
                if (resolved & eSymbolContextLineEntry) m_sc.line_entry = sc.line_entry;
            }
        }
        else
        {
            // If we don't have a module, then we can't have the compile unit,
            // function, block, line entry or symbol, so we can safely call
            // ResolveSymbolContextForAddress with our symbol context member m_sc.
            m_thread.GetProcess().GetTarget().GetImages().ResolveSymbolContextForAddress (lookup_addr, resolve_scope, m_sc);
        }

        // If the target was requested add that:
        if (m_sc.target_sp.get() == NULL)
            m_sc.target_sp = CalculateProcess()->GetTarget().GetSP();

        // Update our internal flags so we remember what we have tried to locate so
        // we don't have to keep trying when more calls to this function are made.
        m_flags.Set(resolve_scope);
    }

    // Return the symbol context with everything that was possible to resolve
    // resolved.
    return m_sc;
}


VariableList *
StackFrame::GetVariableList ()
{
    if (m_flags.IsClear(RESOLVED_VARIABLES))
    {
        m_flags.Set(RESOLVED_VARIABLES);

        GetSymbolContext(eSymbolContextFunction);
        if (m_sc.function)
        {
            bool get_child_variables = true;
            bool can_create = true;
            m_variable_list_sp = m_sc.function->GetBlock (can_create).GetVariableList (get_child_variables, can_create);
        }
    }
    return m_variable_list_sp.get();
}


bool
StackFrame::GetFrameBaseValue (Scalar &frame_base, Error *error_ptr)
{
    if (m_flags.IsClear(GOT_FRAME_BASE))
    {
        if (m_sc.function)
        {
            m_frame_base.Clear();
            m_frame_base_error.Clear();

            m_flags.Set(GOT_FRAME_BASE);
            ExecutionContext exe_ctx (&m_thread.GetProcess(), &m_thread, this);
            Value expr_value;
            if (m_sc.function->GetFrameBaseExpression().Evaluate(&exe_ctx, NULL, NULL, expr_value, &m_frame_base_error) < 0)
            {
                // We should really have an error if evaluate returns, but in case
                // we don't, lets set the error to something at least.
                if (m_frame_base_error.Success())
                    m_frame_base_error.SetErrorString("Evaluation of the frame base expression failed.");
            }
            else
            {
                m_frame_base = expr_value.ResolveValue(&exe_ctx, NULL);
            }
        }
        else
        {
            m_frame_base_error.SetErrorString ("No function in symbol context.");
        }
    }

    if (m_frame_base_error.Success())
        frame_base = m_frame_base;

    if (error_ptr)
        *error_ptr = m_frame_base_error;
    return m_frame_base_error.Success();
}

RegisterContext *
StackFrame::GetRegisterContext ()
{
    if (m_reg_context_sp.get() == NULL)
        m_reg_context_sp.reset (m_thread.CreateRegisterContextForFrame (this));
    return m_reg_context_sp.get();
}

bool
StackFrame::HasDebugInformation ()
{
    GetSymbolContext(eSymbolContextLineEntry);
    return m_sc.line_entry.IsValid();
}

ValueObjectList &
StackFrame::GetValueObjectList()
{
    return m_value_object_list;
}


Target *
StackFrame::CalculateTarget ()
{
    return m_thread.CalculateTarget();
}

Process *
StackFrame::CalculateProcess ()
{
    return m_thread.CalculateProcess();
}

Thread *
StackFrame::CalculateThread ()
{
    return &m_thread;
}

StackFrame *
StackFrame::CalculateStackFrame ()
{
    return this;
}


void
StackFrame::Calculate (ExecutionContext &exe_ctx)
{
    m_thread.Calculate (exe_ctx);
    exe_ctx.frame = this;
}

void
StackFrame::Dump (Stream *strm, bool show_frame_index)
{
    if (strm == NULL)
        return;

    if (show_frame_index)
        strm->Printf("frame #%u: ", m_frame_index);
    strm->Printf("pc = 0x%0*llx", m_thread.GetProcess().GetAddressByteSize() * 2, GetRegisterContext()->GetPC());
    SymbolContext sc (GetSymbolContext(eSymbolContextEverything));
    strm->PutCString(", where = ");
    // TODO: need to get the
    const bool show_module = true;
    const bool show_inline = true;
    sc.DumpStopContext(strm, &m_thread.GetProcess(), GetPC(), show_module, show_inline);
}

