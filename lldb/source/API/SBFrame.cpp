//===-- SBFrame.cpp ---------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/API/SBFrame.h"

#include <string>
#include <algorithm>

#include "lldb/lldb-types.h"

#include "lldb/Core/Address.h"
#include "lldb/Core/ConstString.h"
#include "lldb/Core/Log.h"
#include "lldb/Core/Stream.h"
#include "lldb/Core/StreamFile.h"
#include "lldb/Core/ValueObjectRegister.h"
#include "lldb/Core/ValueObjectVariable.h"
#include "lldb/Expression/ClangUserExpression.h"
#include "lldb/Host/Host.h"
#include "lldb/Symbol/Block.h"
#include "lldb/Symbol/SymbolContext.h"
#include "lldb/Symbol/VariableList.h"
#include "lldb/Symbol/Variable.h"
#include "lldb/Target/ExecutionContext.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/RegisterContext.h"
#include "lldb/Target/StackFrame.h"
#include "lldb/Target/Thread.h"

#include "lldb/API/SBDebugger.h"
#include "lldb/API/SBValue.h"
#include "lldb/API/SBAddress.h"
#include "lldb/API/SBStream.h"
#include "lldb/API/SBSymbolContext.h"
#include "lldb/API/SBThread.h"

using namespace lldb;
using namespace lldb_private;

SBFrame::SBFrame () :
    m_opaque_sp ()
{
}

SBFrame::SBFrame (const StackFrameSP &lldb_object_sp) :
    m_opaque_sp (lldb_object_sp)
{
    LogSP log(GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));

    if (log)
    {
        SBStream sstr;
        GetDescription (sstr);
        log->Printf ("SBFrame::SBFrame (sp=%p) => SBFrame(%p): %s", 
                     lldb_object_sp.get(), m_opaque_sp.get(), sstr.GetData());
                     
    }
}

SBFrame::SBFrame(const SBFrame &rhs) :
    m_opaque_sp (rhs.m_opaque_sp)
{
}

const SBFrame &
SBFrame::operator = (const SBFrame &rhs)
{
    if (this != &rhs)
        m_opaque_sp = rhs.m_opaque_sp;
    return *this;
}

SBFrame::~SBFrame()
{
}


void
SBFrame::SetFrame (const StackFrameSP &lldb_object_sp)
{
    void *old_ptr = m_opaque_sp.get();
    m_opaque_sp = lldb_object_sp;
    LogSP log(GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));

    if (log)
    {
        log->Printf ("SBFrame(%p)::SetFrame(sp=%p) := SBFrame(%p)", 
                     old_ptr, lldb_object_sp.get(), m_opaque_sp.get());
    }

}


bool
SBFrame::IsValid() const
{
    return (m_opaque_sp.get() != NULL);
}

SBSymbolContext
SBFrame::GetSymbolContext (uint32_t resolve_scope) const
{

    SBSymbolContext sb_sym_ctx;
    if (m_opaque_sp)
    {
        Mutex::Locker api_locker (m_opaque_sp->GetThread().GetProcess().GetTarget().GetAPIMutex());
        sb_sym_ctx.SetSymbolContext(&m_opaque_sp->GetSymbolContext (resolve_scope));
    }

    LogSP log(GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    if (log)
        log->Printf ("SBFrame(%p)::GetSymbolContext (resolve_scope=0x%8.8x) => SBSymbolContext(%p)", 
                     m_opaque_sp.get(), resolve_scope, sb_sym_ctx.get());

    return sb_sym_ctx;
}

SBModule
SBFrame::GetModule () const
{
    SBModule sb_module;
    if (m_opaque_sp)
    {
        Mutex::Locker api_locker (m_opaque_sp->GetThread().GetProcess().GetTarget().GetAPIMutex());
        *sb_module = m_opaque_sp->GetSymbolContext (eSymbolContextModule).module_sp;
    }

    LogSP log(GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    if (log)
        log->Printf ("SBFrame(%p)::GetModule () => SBModule(%p)", 
                     m_opaque_sp.get(), sb_module.get());

    return sb_module;
}

SBCompileUnit
SBFrame::GetCompileUnit () const
{
    SBCompileUnit sb_comp_unit;
    if (m_opaque_sp)
    {
        Mutex::Locker api_locker (m_opaque_sp->GetThread().GetProcess().GetTarget().GetAPIMutex());
        sb_comp_unit.reset (m_opaque_sp->GetSymbolContext (eSymbolContextCompUnit).comp_unit);
    }
    LogSP log(GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    if (log)
        log->Printf ("SBFrame(%p)::GetModule () => SBCompileUnit(%p)", 
                     m_opaque_sp.get(), sb_comp_unit.get());

    return sb_comp_unit;
}

SBFunction
SBFrame::GetFunction () const
{
    SBFunction sb_function;
    if (m_opaque_sp)
    {
        Mutex::Locker api_locker (m_opaque_sp->GetThread().GetProcess().GetTarget().GetAPIMutex());
        sb_function.reset(m_opaque_sp->GetSymbolContext (eSymbolContextFunction).function);
    }
    LogSP log(GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    if (log)
        log->Printf ("SBFrame(%p)::GetFunction () => SBFunction(%p)", 
                     m_opaque_sp.get(), sb_function.get());

    return sb_function;
}

SBSymbol
SBFrame::GetSymbol () const
{
    SBSymbol sb_symbol;
    if (m_opaque_sp)
    {
        Mutex::Locker api_locker (m_opaque_sp->GetThread().GetProcess().GetTarget().GetAPIMutex());
        sb_symbol.reset(m_opaque_sp->GetSymbolContext (eSymbolContextSymbol).symbol);
    }
    LogSP log(GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    if (log)
        log->Printf ("SBFrame(%p)::GetSymbol () => SBSymbol(%p)", 
                     m_opaque_sp.get(), sb_symbol.get());
    return sb_symbol;
}

SBBlock
SBFrame::GetBlock () const
{
    SBBlock sb_block;
    if (m_opaque_sp)
    {
        Mutex::Locker api_locker (m_opaque_sp->GetThread().GetProcess().GetTarget().GetAPIMutex());
        sb_block.reset (m_opaque_sp->GetSymbolContext (eSymbolContextBlock).block);
    }
    LogSP log(GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    if (log)
        log->Printf ("SBFrame(%p)::GetBlock () => SBBlock(%p)", 
                     m_opaque_sp.get(), sb_block.get());
    return sb_block;
}

SBBlock
SBFrame::GetFrameBlock () const
{
    SBBlock sb_block;
    if (m_opaque_sp)
    {
        Mutex::Locker api_locker (m_opaque_sp->GetThread().GetProcess().GetTarget().GetAPIMutex());
        sb_block.reset(m_opaque_sp->GetFrameBlock ());
    }
    LogSP log(GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    if (log)
        log->Printf ("SBFrame(%p)::GetFrameBlock () => SBBlock(%p)", 
                     m_opaque_sp.get(), sb_block.get());
    return sb_block;    
}

SBLineEntry
SBFrame::GetLineEntry () const
{
    SBLineEntry sb_line_entry;
    if (m_opaque_sp)
    {
        Mutex::Locker api_locker (m_opaque_sp->GetThread().GetProcess().GetTarget().GetAPIMutex());
        sb_line_entry.SetLineEntry (m_opaque_sp->GetSymbolContext (eSymbolContextLineEntry).line_entry);
    }
    LogSP log(GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    if (log)
        log->Printf ("SBFrame(%p)::GetLineEntry () => SBLineEntry(%p)", 
                     m_opaque_sp.get(), sb_line_entry.get());
    return sb_line_entry;
}

uint32_t
SBFrame::GetFrameID () const
{
    uint32_t frame_idx = m_opaque_sp ? m_opaque_sp->GetFrameIndex () : UINT32_MAX;
    
    LogSP log(GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    if (log)
        log->Printf ("SBFrame(%p)::GetFrameID () => %u", 
                     m_opaque_sp.get(), frame_idx);
    return frame_idx;
}

addr_t
SBFrame::GetPC () const
{
    addr_t addr = LLDB_INVALID_ADDRESS;
    if (m_opaque_sp)
    {
        Mutex::Locker api_locker (m_opaque_sp->GetThread().GetProcess().GetTarget().GetAPIMutex());
        addr = m_opaque_sp->GetFrameCodeAddress().GetOpcodeLoadAddress (&m_opaque_sp->GetThread().GetProcess().GetTarget());
    }

    LogSP log(GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    if (log)
        log->Printf ("SBFrame(%p)::GetPC () => 0x%llx", m_opaque_sp.get(), addr);

    return addr;
}

bool
SBFrame::SetPC (addr_t new_pc)
{
    bool ret_val = false;
    if (m_opaque_sp)
    {
        Mutex::Locker api_locker (m_opaque_sp->GetThread().GetProcess().GetTarget().GetAPIMutex());
        ret_val = m_opaque_sp->GetRegisterContext()->SetPC (new_pc);
    }

    LogSP log(GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    if (log)
        log->Printf ("SBFrame(%p)::SetPC (new_pc=0x%llx) => %i", 
                     m_opaque_sp.get(), new_pc, ret_val);

    return ret_val;
}

addr_t
SBFrame::GetSP () const
{
    addr_t addr = LLDB_INVALID_ADDRESS;
    if (m_opaque_sp)
    {
        Mutex::Locker api_locker (m_opaque_sp->GetThread().GetProcess().GetTarget().GetAPIMutex());
        addr = m_opaque_sp->GetRegisterContext()->GetSP();
    }
    LogSP log(GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    if (log)
        log->Printf ("SBFrame(%p)::GetSP () => 0x%llx", m_opaque_sp.get(), addr);

    return addr;
}


addr_t
SBFrame::GetFP () const
{
    addr_t addr = LLDB_INVALID_ADDRESS;
    if (m_opaque_sp)
    {
        Mutex::Locker api_locker (m_opaque_sp->GetThread().GetProcess().GetTarget().GetAPIMutex());
        addr = m_opaque_sp->GetRegisterContext()->GetFP();
    }

    LogSP log(GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    if (log)
        log->Printf ("SBFrame(%p)::GetFP () => 0x%llx", m_opaque_sp.get(), addr);
    return addr;
}


SBAddress
SBFrame::GetPCAddress () const
{
    SBAddress sb_addr;
    if (m_opaque_sp)
    {
        Mutex::Locker api_locker (m_opaque_sp->GetThread().GetProcess().GetTarget().GetAPIMutex());
        sb_addr.SetAddress (&m_opaque_sp->GetFrameCodeAddress());
    }
    LogSP log(GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    if (log)
        log->Printf ("SBFrame(%p)::GetPCAddress () => SBAddress(%p)", m_opaque_sp.get(), sb_addr.get());
    return sb_addr;
}

void
SBFrame::Clear()
{
    m_opaque_sp.reset();
}

SBValue
SBFrame::FindVariable (const char *name)
{
    SBValue value;
    if (m_opaque_sp)
    {
        lldb::DynamicValueType  use_dynamic = m_opaque_sp->CalculateTarget()->GetPreferDynamicValue();
        value = FindVariable (name, use_dynamic);
    }
    return value;
}

SBValue
SBFrame::FindVariable (const char *name, lldb::DynamicValueType use_dynamic)
{
    VariableSP var_sp;
    SBValue sb_value;
    
    if (m_opaque_sp && name && name[0])
    {
        VariableList variable_list;
        Mutex::Locker api_locker (m_opaque_sp->GetThread().GetProcess().GetTarget().GetAPIMutex());
        SymbolContext sc (m_opaque_sp->GetSymbolContext (eSymbolContextBlock));

        if (sc.block)
        {
            const bool can_create = true;
            const bool get_parent_variables = true;
            const bool stop_if_block_is_inlined_function = true;
            
            if (sc.block->AppendVariables (can_create, 
                                           get_parent_variables,
                                           stop_if_block_is_inlined_function,
                                           &variable_list))
            {
                var_sp = variable_list.FindVariable (ConstString(name));
            }
        }

        if (var_sp)
            *sb_value = ValueObjectSP (m_opaque_sp->GetValueObjectForFrameVariable(var_sp, use_dynamic));
        
    }
    
    LogSP log(GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    if (log)
        log->Printf ("SBFrame(%p)::FindVariable (name=\"%s\") => SBValue(%p)", 
                     m_opaque_sp.get(), name, sb_value.get());

    return sb_value;
}

SBValue
SBFrame::FindValue (const char *name, ValueType value_type)
{
    SBValue value;
    if (m_opaque_sp)
    {
        lldb::DynamicValueType use_dynamic = m_opaque_sp->CalculateTarget()->GetPreferDynamicValue();
        value = FindValue (name, value_type, use_dynamic);
    }
    return value;
}

SBValue
SBFrame::FindValue (const char *name, ValueType value_type, lldb::DynamicValueType use_dynamic)
{
    SBValue sb_value;
    if (m_opaque_sp && name && name[0])
    {
        Mutex::Locker api_locker (m_opaque_sp->GetThread().GetProcess().GetTarget().GetAPIMutex());
    
        switch (value_type)
        {
        case eValueTypeVariableGlobal:      // global variable
        case eValueTypeVariableStatic:      // static variable
        case eValueTypeVariableArgument:    // function argument variables
        case eValueTypeVariableLocal:       // function local variables
            {
                VariableList *variable_list = m_opaque_sp->GetVariableList(true);

                SymbolContext sc (m_opaque_sp->GetSymbolContext (eSymbolContextBlock));

                const bool can_create = true;
                const bool get_parent_variables = true;
                const bool stop_if_block_is_inlined_function = true;

                if (sc.block && sc.block->AppendVariables (can_create, 
                                                           get_parent_variables,
                                                           stop_if_block_is_inlined_function,
                                                           variable_list))
                {
                    ConstString const_name(name);
                    const uint32_t num_variables = variable_list->GetSize();
                    for (uint32_t i = 0; i < num_variables; ++i)
                    {
                        VariableSP variable_sp (variable_list->GetVariableAtIndex(i));
                        if (variable_sp && 
                            variable_sp->GetScope() == value_type &&
                            variable_sp->GetName() == const_name)
                        {
                            *sb_value = ValueObjectSP (m_opaque_sp->GetValueObjectForFrameVariable(variable_sp, 
                                                                                                   use_dynamic));
                            break;
                        }
                    }
                }
            }
            break;

        case eValueTypeRegister:            // stack frame register value
            {
                RegisterContextSP reg_ctx (m_opaque_sp->GetRegisterContext());
                if (reg_ctx)
                {
                    const uint32_t num_regs = reg_ctx->GetRegisterCount();
                    for (uint32_t reg_idx = 0; reg_idx < num_regs; ++reg_idx)
                    {
                        const RegisterInfo *reg_info = reg_ctx->GetRegisterInfoAtIndex (reg_idx);
                        if (reg_info && 
                            ((reg_info->name && strcasecmp (reg_info->name, name) == 0) ||
                             (reg_info->alt_name && strcasecmp (reg_info->alt_name, name) == 0)))
                        {
                            *sb_value = ValueObjectRegister::Create (m_opaque_sp.get(), reg_ctx, reg_idx);
                        }
                    }
                }
            }
            break;

        case eValueTypeRegisterSet:         // A collection of stack frame register values
            {
                RegisterContextSP reg_ctx (m_opaque_sp->GetRegisterContext());
                if (reg_ctx)
                {
                    const uint32_t num_sets = reg_ctx->GetRegisterSetCount();
                    for (uint32_t set_idx = 0; set_idx < num_sets; ++set_idx)
                    {
                        const RegisterSet *reg_set = reg_ctx->GetRegisterSet (set_idx);
                        if (reg_set && 
                            ((reg_set->name && strcasecmp (reg_set->name, name) == 0) ||
                             (reg_set->short_name && strcasecmp (reg_set->short_name, name) == 0)))
                        {
                            *sb_value = ValueObjectRegisterSet::Create (m_opaque_sp.get(), reg_ctx, set_idx);
                        }
                    }
                }
            }
            break;

        case eValueTypeConstResult:         // constant result variables
            {
                ConstString const_name(name);
                ClangExpressionVariableSP expr_var_sp (m_opaque_sp->GetThread().GetProcess().GetTarget().GetPersistentVariables().GetVariable (const_name));
                if (expr_var_sp)
                    *sb_value = expr_var_sp->GetValueObject();
            }
            break;

        default:
            break;
        }
    }
    
    LogSP log(GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    if (log)
        log->Printf ("SBFrame(%p)::FindVariableInScope (name=\"%s\", value_type=%i) => SBValue(%p)", 
                     m_opaque_sp.get(), name, value_type, sb_value.get());

    
    return sb_value;
}

bool
SBFrame::operator == (const SBFrame &rhs) const
{
    return m_opaque_sp.get() == rhs.m_opaque_sp.get();
}

bool
SBFrame::operator != (const SBFrame &rhs) const
{
    return m_opaque_sp.get() != rhs.m_opaque_sp.get();
}

lldb_private::StackFrame *
SBFrame::operator->() const
{
    return m_opaque_sp.get();
}

lldb_private::StackFrame *
SBFrame::get() const
{
    return m_opaque_sp.get();
}

const lldb::StackFrameSP &
SBFrame::get_sp() const
{
    return m_opaque_sp;
}

SBThread
SBFrame::GetThread () const
{
    LogSP log(GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));

    SBThread sb_thread;
    if (m_opaque_sp)
    {
        Mutex::Locker api_locker (m_opaque_sp->GetThread().GetProcess().GetTarget().GetAPIMutex());
        sb_thread.SetThread (m_opaque_sp->GetThread().GetSP());
    }

    if (log)
    {
        SBStream sstr;
        sb_thread.GetDescription (sstr);
        log->Printf ("SBFrame(%p)::GetThread () => SBThread(%p): %s", m_opaque_sp.get(), 
                     sb_thread.get(), sstr.GetData());
    }

    return sb_thread;
}

const char *
SBFrame::Disassemble () const
{
    const char *disassembly = NULL;
    if (m_opaque_sp)
    {
        Mutex::Locker api_locker (m_opaque_sp->GetThread().GetProcess().GetTarget().GetAPIMutex());
        disassembly = m_opaque_sp->Disassemble();
    }
    LogSP log(GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));

    if (log)
        log->Printf ("SBFrame(%p)::Disassemble () => %s", m_opaque_sp.get(), disassembly);

    return disassembly;
}


SBValueList
SBFrame::GetVariables (bool arguments,
                       bool locals,
                       bool statics,
                       bool in_scope_only)
{
    SBValueList value_list;
    if (m_opaque_sp)
    {
        lldb::DynamicValueType use_dynamic = m_opaque_sp->CalculateTarget()->GetPreferDynamicValue();
        value_list = GetVariables (arguments, locals, statics, in_scope_only, use_dynamic);
    }
    return value_list;
}

SBValueList
SBFrame::GetVariables (bool arguments,
                       bool locals,
                       bool statics,
                       bool in_scope_only,
                       lldb::DynamicValueType  use_dynamic)
{
    LogSP log(GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));

    if (log)
        log->Printf ("SBFrame(%p)::GetVariables (arguments=%i, locals=%i, statics=%i, in_scope_only=%i)", 
                     m_opaque_sp.get(), 
                     arguments,
                     locals,
                     statics,
                     in_scope_only);

    SBValueList value_list;
    if (m_opaque_sp)
    {

        size_t i;
        VariableList *variable_list = NULL;
        // Scope for locker
        {
            Mutex::Locker api_locker (m_opaque_sp->GetThread().GetProcess().GetTarget().GetAPIMutex());
            variable_list = m_opaque_sp->GetVariableList(true);
        }
        if (variable_list)
        {
            const size_t num_variables = variable_list->GetSize();
            if (num_variables)
            {
                for (i = 0; i < num_variables; ++i)
                {
                    VariableSP variable_sp (variable_list->GetVariableAtIndex(i));
                    if (variable_sp)
                    {
                        bool add_variable = false;
                        switch (variable_sp->GetScope())
                        {
                        case eValueTypeVariableGlobal:
                        case eValueTypeVariableStatic:
                            add_variable = statics;
                            break;

                        case eValueTypeVariableArgument:
                            add_variable = arguments;
                            break;

                        case eValueTypeVariableLocal:
                            add_variable = locals;
                            break;

                        default:
                            break;
                        }
                        if (add_variable)
                        {
                            if (in_scope_only && !variable_sp->IsInScope(m_opaque_sp.get()))
                                continue;

                            value_list.Append(m_opaque_sp->GetValueObjectForFrameVariable (variable_sp, use_dynamic));
                        }
                    }
                }
            }
        }        
    }

    if (log)
    {
        log->Printf ("SBFrame(%p)::GetVariables (...) => SBValueList(%p)", m_opaque_sp.get(),
                     value_list.get());
    }

    return value_list;
}

SBValueList
SBFrame::GetRegisters ()
{
    LogSP log(GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));

    SBValueList value_list;
    if (m_opaque_sp)
    {
        Mutex::Locker api_locker (m_opaque_sp->GetThread().GetProcess().GetTarget().GetAPIMutex());
        RegisterContextSP reg_ctx (m_opaque_sp->GetRegisterContext());
        if (reg_ctx)
        {
            const uint32_t num_sets = reg_ctx->GetRegisterSetCount();
            for (uint32_t set_idx = 0; set_idx < num_sets; ++set_idx)
            {
                value_list.Append(ValueObjectRegisterSet::Create (m_opaque_sp.get(), reg_ctx, set_idx));
            }
        }
    }

    if (log)
        log->Printf ("SBFrame(%p)::Registers () => SBValueList(%p)", m_opaque_sp.get(), value_list.get());

    return value_list;
}

bool
SBFrame::GetDescription (SBStream &description)
{
    if (m_opaque_sp)
    {
        Mutex::Locker api_locker (m_opaque_sp->GetThread().GetProcess().GetTarget().GetAPIMutex());
        Stream &s = description.ref();
        m_opaque_sp->DumpUsingSettingsFormat (&s);
    }
    else
        description.Printf ("No value");

    return true;
}

SBValue
SBFrame::EvaluateExpression (const char *expr)
{
    SBValue result;
    if (m_opaque_sp)
    {
        lldb::DynamicValueType use_dynamic = m_opaque_sp->CalculateTarget()->GetPreferDynamicValue();
        result = EvaluateExpression (expr, use_dynamic);
    }
    return result;
}

SBValue
SBFrame::EvaluateExpression (const char *expr, lldb::DynamicValueType fetch_dynamic_value)
{
    LogSP log(GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    
    LogSP expr_log(GetLogIfAllCategoriesSet (LIBLLDB_LOG_EXPRESSIONS));

    ExecutionResults exe_results;
    SBValue expr_result;
    if (log)
        log->Printf ("SBFrame(%p)::EvaluateExpression (expr=\"%s\")...", m_opaque_sp.get(), expr);

    if (m_opaque_sp)
    {
        Mutex::Locker api_locker (m_opaque_sp->GetThread().GetProcess().GetTarget().GetAPIMutex());
        
        
        StreamString frame_description;
        m_opaque_sp->DumpUsingSettingsFormat (&frame_description);

        Host::SetCrashDescriptionWithFormat ("SBFrame::EvaluateExpression (expr = \"%s\", fetch_dynamic_value = %u) %s",
                                             expr, fetch_dynamic_value, frame_description.GetString().c_str());

        const bool unwind_on_error = true;
        const bool keep_in_memory = false;

        exe_results = m_opaque_sp->GetThread().GetProcess().GetTarget().EvaluateExpression(expr, 
                                                                                           m_opaque_sp.get(), 
                                                                                           unwind_on_error, 
                                                                                           keep_in_memory, 
                                                                                           fetch_dynamic_value, 
                                                                                           *expr_result);
    }
    
    if (expr_log)
        expr_log->Printf("** [SBFrame::EvaluateExpression] Expression result is %s, summary %s **", 
                         expr_result.GetValue(), 
                         expr_result.GetSummary());
    
    if (log)
        log->Printf ("SBFrame(%p)::EvaluateExpression (expr=\"%s\") => SBValue(%p) (execution result=%d)", m_opaque_sp.get(), 
                     expr, 
                     expr_result.get(),
                     exe_results);

    return expr_result;
}

bool
SBFrame::IsInlined()
{
    if (m_opaque_sp)
    {
        Block *block = m_opaque_sp->GetSymbolContext(eSymbolContextBlock).block;
        if (block)
            return block->GetContainingInlinedBlock () != NULL;
    }
    return false;
}

const char *
SBFrame::GetFunctionName()
{
    const char *name = NULL;
    if (m_opaque_sp)
    {
        SymbolContext sc (m_opaque_sp->GetSymbolContext(eSymbolContextFunction | eSymbolContextBlock | eSymbolContextSymbol));
        if (sc.block)
        {
            Block *inlined_block = sc.block->GetContainingInlinedBlock ();
            if (inlined_block)
            {
                const InlineFunctionInfo* inlined_info = inlined_block->GetInlinedFunctionInfo();
                name = inlined_info->GetName().AsCString();
            }
        }
        
        if (name == NULL)
        {
            if (sc.function)
                name = sc.function->GetName().GetCString();
        }

        if (name == NULL)
        {
            if (sc.symbol)
                name = sc.symbol->GetName().GetCString();
        }
    }
    return name;
}

