//===-- ClangExpressionDeclMap.cpp -----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Expression/ClangExpressionDeclMap.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclarationName.h"
#include "clang/AST/Decl.h"
#include "lldb/lldb-private.h"
#include "lldb/Core/Address.h"
#include "lldb/Core/Error.h"
#include "lldb/Core/Log.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/RegisterValue.h"
#include "lldb/Core/ValueObjectConstResult.h"
#include "lldb/Core/ValueObjectVariable.h"
#include "lldb/Expression/ASTDumper.h"
#include "lldb/Expression/ClangASTSource.h"
#include "lldb/Expression/ClangPersistentVariables.h"
#include "lldb/Expression/Materializer.h"
#include "lldb/Host/Endian.h"
#include "lldb/Symbol/ClangASTContext.h"
#include "lldb/Symbol/ClangNamespaceDecl.h"
#include "lldb/Symbol/CompileUnit.h"
#include "lldb/Symbol/Function.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Symbol/SymbolContext.h"
#include "lldb/Symbol/SymbolVendor.h"
#include "lldb/Symbol/Type.h"
#include "lldb/Symbol/TypeList.h"
#include "lldb/Symbol/Variable.h"
#include "lldb/Symbol/VariableList.h"
#include "lldb/Target/ExecutionContext.h"
#include "lldb/Target/ObjCLanguageRuntime.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/RegisterContext.h"
#include "lldb/Target/StackFrame.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/Thread.h"

using namespace lldb;
using namespace lldb_private;
using namespace clang;

ClangExpressionDeclMap::ClangExpressionDeclMap (bool keep_result_in_memory, ExecutionContext &exe_ctx) :
    ClangASTSource (exe_ctx.GetTargetSP()),
    m_found_entities (),
    m_struct_members (),
    m_keep_result_in_memory (keep_result_in_memory),
    m_parser_vars (),
    m_struct_vars ()
{
    EnableStructVars();
}

ClangExpressionDeclMap::~ClangExpressionDeclMap()
{
    // Note: The model is now that the parser's AST context and all associated
    //   data does not vanish until the expression has been executed.  This means
    //   that valuable lookup data (like namespaces) doesn't vanish, but 
    
    DidParse();
    DidDematerialize();
    DisableStructVars();
}

bool 
ClangExpressionDeclMap::WillParse(ExecutionContext &exe_ctx,
                                  Materializer *materializer)
{
    ClangASTMetrics::ClearLocalCounters();
    
    EnableParserVars();
    m_parser_vars->m_exe_ctx = exe_ctx;
    
    Target *target = exe_ctx.GetTargetPtr();
    if (exe_ctx.GetFramePtr())
        m_parser_vars->m_sym_ctx = exe_ctx.GetFramePtr()->GetSymbolContext(lldb::eSymbolContextEverything);
    else if (exe_ctx.GetThreadPtr() && exe_ctx.GetThreadPtr()->GetStackFrameAtIndex(0))
        m_parser_vars->m_sym_ctx = exe_ctx.GetThreadPtr()->GetStackFrameAtIndex(0)->GetSymbolContext(lldb::eSymbolContextEverything);
    else if (exe_ctx.GetProcessPtr())
    {
        m_parser_vars->m_sym_ctx.Clear(true);
        m_parser_vars->m_sym_ctx.target_sp = exe_ctx.GetTargetSP();
    }
    else if (target)
    {
        m_parser_vars->m_sym_ctx.Clear(true);
        m_parser_vars->m_sym_ctx.target_sp = exe_ctx.GetTargetSP();
    }
    
    if (target)
    {
        m_parser_vars->m_persistent_vars = &target->GetPersistentVariables();
    
        if (!target->GetScratchClangASTContext())
            return false;
    }
    
    m_parser_vars->m_target_info = GetTargetInfo();
    m_parser_vars->m_materializer = materializer;
    
    return true;
}

void
ClangExpressionDeclMap::DidParse()
{
    Log *log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_EXPRESSIONS));

    if (log)
        ClangASTMetrics::DumpCounters(log);
    
    if (m_parser_vars.get())
    {
        for (size_t entity_index = 0, num_entities = m_found_entities.GetSize();
             entity_index < num_entities;
             ++entity_index)
        {
            ClangExpressionVariableSP var_sp(m_found_entities.GetVariableAtIndex(entity_index));
            if (var_sp)
            {
                ClangExpressionVariable::ParserVars *parser_vars = var_sp->GetParserVars(GetParserID());
                
                if (parser_vars && parser_vars->m_lldb_value)
                    delete parser_vars->m_lldb_value;
            
                var_sp->DisableParserVars(GetParserID());
            }
        }
        
        for (size_t pvar_index = 0, num_pvars = m_parser_vars->m_persistent_vars->GetSize();
             pvar_index < num_pvars;
             ++pvar_index)
        {
            ClangExpressionVariableSP pvar_sp(m_parser_vars->m_persistent_vars->GetVariableAtIndex(pvar_index));
            if (pvar_sp)
                pvar_sp->DisableParserVars(GetParserID());
        }
        
        DisableParserVars();
    }
}

// Interface for IRForTarget

ClangExpressionDeclMap::TargetInfo 
ClangExpressionDeclMap::GetTargetInfo()
{
    assert (m_parser_vars.get());
    
    TargetInfo ret;
    
    ExecutionContext &exe_ctx = m_parser_vars->m_exe_ctx;

    Process *process = exe_ctx.GetProcessPtr();
    if (process)
    {
        ret.byte_order = process->GetByteOrder();
        ret.address_byte_size = process->GetAddressByteSize();
    }
    else 
    {
        Target *target = exe_ctx.GetTargetPtr();
        if (target)
        {
            ret.byte_order = target->GetArchitecture().GetByteOrder();
            ret.address_byte_size = target->GetArchitecture().GetAddressByteSize();
        }
    }

    return ret;
}

const ConstString &
ClangExpressionDeclMap::GetPersistentResultName ()
{
    assert (m_struct_vars.get());
    assert (m_parser_vars.get());
    if (!m_struct_vars->m_result_name)
    {
        Target *target = m_parser_vars->GetTarget();
        assert (target);
        m_struct_vars->m_result_name = target->GetPersistentVariables().GetNextPersistentVariableName();
    }
    return m_struct_vars->m_result_name;
}

lldb::ClangExpressionVariableSP
ClangExpressionDeclMap::BuildIntegerVariable (const ConstString &name,
                                              lldb_private::TypeFromParser type,
                                              const llvm::APInt& value)
{
    assert (m_parser_vars.get());
    
    ExecutionContext &exe_ctx = m_parser_vars->m_exe_ctx;
    
    Target *target = exe_ctx.GetTargetPtr();
    
    if (!target)
        return ClangExpressionVariableSP();

    ASTContext *context(target->GetScratchClangASTContext()->getASTContext());
    
    TypeFromUser user_type(m_ast_importer->CopyType(context, 
                                                    type.GetASTContext(),
                                                    type.GetOpaqueQualType()),
                           context);
    
    if (!user_type.GetOpaqueQualType())
    {
        Log *log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_EXPRESSIONS));

        if (log)
            log->Printf("ClangExpressionDeclMap::BuildIntegerVariable - Couldn't export the type for a constant integer result");
        
        return lldb::ClangExpressionVariableSP();
    }
    
    if (!m_parser_vars->m_persistent_vars->CreatePersistentVariable (exe_ctx.GetBestExecutionContextScope (),
                                                                     name, 
                                                                     user_type, 
                                                                     m_parser_vars->m_target_info.byte_order,
                                                                     m_parser_vars->m_target_info.address_byte_size))
        return lldb::ClangExpressionVariableSP();
    
    ClangExpressionVariableSP pvar_sp (m_parser_vars->m_persistent_vars->GetVariable(name));
    
    if (!pvar_sp)
        return lldb::ClangExpressionVariableSP();
    
    uint8_t *pvar_data = pvar_sp->GetValueBytes();
    if (pvar_data == NULL)
        return lldb::ClangExpressionVariableSP();
    
    uint64_t value64 = value.getLimitedValue();
        
    size_t num_val_bytes = sizeof(value64);
    size_t num_data_bytes = pvar_sp->GetByteSize();
    
    size_t num_bytes = num_val_bytes;
    if (num_bytes > num_data_bytes)
        num_bytes = num_data_bytes;
    
    for (size_t byte_idx = 0;
         byte_idx < num_bytes;
         ++byte_idx)
    {
        uint64_t shift = byte_idx * 8;
        uint64_t mask = 0xffll << shift;
        uint8_t cur_byte = (uint8_t)((value64 & mask) >> shift);
        
        switch (m_parser_vars->m_target_info.byte_order)
        {
            case eByteOrderBig:
                //                    High         Low
                // Original:         |AABBCCDDEEFFGGHH|
                // Target:                   |EEFFGGHH|
                
                pvar_data[num_data_bytes - (1 + byte_idx)] = cur_byte;
                break;
            case eByteOrderLittle:
                // Target:                   |HHGGFFEE|
                pvar_data[byte_idx] = cur_byte;
                break;
            default:
                return lldb::ClangExpressionVariableSP();    
        }
    }
    
    pvar_sp->m_flags |= ClangExpressionVariable::EVIsFreezeDried;
    pvar_sp->m_flags |= ClangExpressionVariable::EVIsLLDBAllocated;
    pvar_sp->m_flags |= ClangExpressionVariable::EVNeedsAllocation;

    return pvar_sp;
}

lldb::ClangExpressionVariableSP
ClangExpressionDeclMap::BuildCastVariable (const ConstString &name,
                                           VarDecl *decl,
                                           lldb_private::TypeFromParser type)
{
    assert (m_parser_vars.get());
    
    Log *log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_EXPRESSIONS));
    
    ExecutionContext &exe_ctx = m_parser_vars->m_exe_ctx;
    Target *target = exe_ctx.GetTargetPtr();
    if (target == NULL)
        return lldb::ClangExpressionVariableSP();

    ASTContext *context(target->GetScratchClangASTContext()->getASTContext());
    
    ClangExpressionVariableSP var_sp (m_found_entities.GetVariable(decl, GetParserID()));
    
    if (!var_sp)
        var_sp = m_parser_vars->m_persistent_vars->GetVariable(decl, GetParserID());
    
    if (!var_sp)
        return ClangExpressionVariableSP();
    
    TypeFromUser user_type(m_ast_importer->CopyType(context, 
                                                    type.GetASTContext(),
                                                    type.GetOpaqueQualType()),
                           context);
    
    if (!user_type.GetOpaqueQualType())
    {        
        if (log)
            log->Printf("ClangExpressionDeclMap::BuildCastVariable - Couldn't export the type for a constant cast result");
        
        return lldb::ClangExpressionVariableSP();
    }
    
    TypeFromUser var_type = var_sp->GetTypeFromUser();
    
    StackFrame *frame = exe_ctx.GetFramePtr();
    if (frame == NULL)
        return lldb::ClangExpressionVariableSP();
    
    VariableSP var = FindVariableInScope (*frame, var_sp->GetName(), &var_type);
    
    if (!var)
        return lldb::ClangExpressionVariableSP(); // but we should handle this; it may be a persistent variable
    
    ValueObjectSP var_valobj = frame->GetValueObjectForFrameVariable(var, lldb::eNoDynamicValues);

    if (!var_valobj)
        return lldb::ClangExpressionVariableSP();
    
    ValueObjectSP var_casted_valobj = var_valobj->CastPointerType(name.GetCString(), user_type);
    
    if (!var_casted_valobj)
        return lldb::ClangExpressionVariableSP();
    
    if (log)
    {
        StreamString my_stream_string;
        
        ClangASTType::DumpTypeDescription (var_type.GetASTContext(),
                                           var_type.GetOpaqueQualType(),
                                           &my_stream_string);
        
        
        log->Printf("Building cast variable to type: %s", my_stream_string.GetString().c_str());
    }
    
    ClangExpressionVariableSP pvar_sp = m_parser_vars->m_persistent_vars->CreatePersistentVariable (var_casted_valobj);
    
    if (!pvar_sp)
        return lldb::ClangExpressionVariableSP();
    
    if (pvar_sp != m_parser_vars->m_persistent_vars->GetVariable(name))
        return lldb::ClangExpressionVariableSP();
    
    pvar_sp->m_flags |= ClangExpressionVariable::EVIsFreezeDried;
    pvar_sp->m_flags |= ClangExpressionVariable::EVIsLLDBAllocated;
    pvar_sp->m_flags |= ClangExpressionVariable::EVNeedsAllocation;
            
    return pvar_sp;
}

bool
ClangExpressionDeclMap::ResultIsReference (const ConstString &name)
{
    ClangExpressionVariableSP pvar_sp = m_parser_vars->m_persistent_vars->GetVariable(name);
    
    return (pvar_sp->m_flags & ClangExpressionVariable::EVIsProgramReference);
}

bool
ClangExpressionDeclMap::CompleteResultVariable (lldb::ClangExpressionVariableSP &valobj,
                                                IRMemoryMap &map,
                                                lldb_private::Value &value,
                                                const ConstString &name,
                                                lldb_private::TypeFromParser type,
                                                bool transient,
                                                bool maybe_make_load)
{
    assert (m_parser_vars.get());
        
    ClangExpressionVariableSP pvar_sp = m_parser_vars->m_persistent_vars->GetVariable(name);
    
    if (!pvar_sp)
        return false;
        
    if (maybe_make_load && 
        value.GetValueType() == Value::eValueTypeFileAddress &&
        m_parser_vars->m_exe_ctx.GetProcessPtr())
    {
        value.SetValueType(Value::eValueTypeLoadAddress);
    }
    
    if (pvar_sp->m_flags & ClangExpressionVariable::EVIsProgramReference &&
        !pvar_sp->m_live_sp &&
        !transient)
    {
        // The reference comes from the program.  We need to set up a live SP for it.
        
        unsigned long long address = value.GetScalar().ULongLong();
        AddressType address_type = value.GetValueAddressType();
        
        pvar_sp->m_live_sp = ValueObjectConstResult::Create(m_parser_vars->m_exe_ctx.GetBestExecutionContextScope(),
                                                            pvar_sp->GetTypeFromUser().GetASTContext(),
                                                            pvar_sp->GetTypeFromUser().GetOpaqueQualType(),
                                                            pvar_sp->GetName(),
                                                            address,
                                                            address_type,
                                                            pvar_sp->GetByteSize());
    }
    
    if (pvar_sp->m_flags & ClangExpressionVariable::EVNeedsFreezeDry)
    {
        pvar_sp->ValueUpdated();
        
        const size_t pvar_byte_size = pvar_sp->GetByteSize();
        uint8_t *pvar_data = pvar_sp->GetValueBytes();
        
        if (!ReadTarget(map, pvar_data, value, pvar_byte_size))
            return false;
        
        pvar_sp->m_flags &= ~(ClangExpressionVariable::EVNeedsFreezeDry);
    }
    
    valobj = pvar_sp;
    
    return true;
}

void
ClangExpressionDeclMap::RemoveResultVariable
(
    const ConstString &name
)
{
    ClangExpressionVariableSP pvar_sp = m_parser_vars->m_persistent_vars->GetVariable(name);
    m_parser_vars->m_persistent_vars->RemovePersistentVariable(pvar_sp);
}

bool 
ClangExpressionDeclMap::AddPersistentVariable 
(
    const NamedDecl *decl, 
    const ConstString &name, 
    TypeFromParser parser_type,
    bool is_result,
    bool is_lvalue
)
{
    assert (m_parser_vars.get());
    
    if (m_parser_vars->m_materializer && is_result)
    {
        Error err;
        
        ExecutionContext &exe_ctx = m_parser_vars->m_exe_ctx;
        Target *target = exe_ctx.GetTargetPtr();
        if (target == NULL)
            return false;
        
        ASTContext *context(target->GetScratchClangASTContext()->getASTContext());
        
        TypeFromUser user_type(m_ast_importer->DeportType(context,
                                                          parser_type.GetASTContext(),
                                                          parser_type.GetOpaqueQualType()),
                               context);
        
        uint32_t offset = m_parser_vars->m_materializer->AddResultVariable(user_type, is_lvalue, m_keep_result_in_memory, err);
        
        m_found_entities.CreateVariable(exe_ctx.GetBestExecutionContextScope(),
                                        name,
                                        user_type,
                                        m_parser_vars->m_target_info.byte_order,
                                        m_parser_vars->m_target_info.address_byte_size);
        
        ClangExpressionVariableSP var_sp (m_found_entities.GetVariable(name));
        
        if (!var_sp)
            return false;
        
        var_sp->EnableParserVars(GetParserID());
        
        ClangExpressionVariable::ParserVars *parser_vars = var_sp->GetParserVars(GetParserID());

        parser_vars->m_named_decl = decl;
        parser_vars->m_parser_type = parser_type;
        
        var_sp->EnableJITVars(GetParserID());
        
        ClangExpressionVariable::JITVars *jit_vars = var_sp->GetJITVars(GetParserID());
        
        jit_vars->m_offset = offset;
        
        return true;
    }
    
    Log *log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_EXPRESSIONS));
    ExecutionContext &exe_ctx = m_parser_vars->m_exe_ctx;
    Target *target = exe_ctx.GetTargetPtr();
    if (target == NULL)
        return false;

    ASTContext *context(target->GetScratchClangASTContext()->getASTContext());
    
    TypeFromUser user_type(m_ast_importer->DeportType(context, 
                                                      parser_type.GetASTContext(),
                                                      parser_type.GetOpaqueQualType()),
                           context);
    
    if (!user_type.GetOpaqueQualType())
    {
        if (log)
            log->Printf("Persistent variable's type wasn't copied successfully");
        return false;
    }
        
    if (!m_parser_vars->m_target_info.IsValid())
        return false;
    
    if (!m_parser_vars->m_persistent_vars->CreatePersistentVariable (exe_ctx.GetBestExecutionContextScope (),
                                                                     name, 
                                                                     user_type, 
                                                                     m_parser_vars->m_target_info.byte_order,
                                                                     m_parser_vars->m_target_info.address_byte_size))
        return false;
    
    ClangExpressionVariableSP var_sp (m_parser_vars->m_persistent_vars->GetVariable(name));
    
    if (!var_sp)
        return false;
    
    var_sp->m_frozen_sp->SetHasCompleteType();
    
    if (is_result)
        var_sp->m_flags |= ClangExpressionVariable::EVNeedsFreezeDry;
    else
        var_sp->m_flags |= ClangExpressionVariable::EVKeepInTarget; // explicitly-declared persistent variables should persist
    
    if (is_lvalue)
    {
        var_sp->m_flags |= ClangExpressionVariable::EVIsProgramReference;
    }
    else
    {
        var_sp->m_flags |= ClangExpressionVariable::EVIsLLDBAllocated;
        var_sp->m_flags |= ClangExpressionVariable::EVNeedsAllocation;
    }
    
    if (m_keep_result_in_memory)
    {
        var_sp->m_flags |= ClangExpressionVariable::EVKeepInTarget;
    }
    
    if (log)
        log->Printf("Created persistent variable with flags 0x%hx", var_sp->m_flags);
    
    var_sp->EnableParserVars(GetParserID());
    
    ClangExpressionVariable::ParserVars *parser_vars = var_sp->GetParserVars(GetParserID());
    
    parser_vars->m_named_decl = decl;
    parser_vars->m_parser_type = parser_type;
    
    return true;
}

bool 
ClangExpressionDeclMap::AddValueToStruct 
(
    const NamedDecl *decl,
    const ConstString &name,
    llvm::Value *value,
    size_t size,
    off_t alignment
)
{
    assert (m_struct_vars.get());
    assert (m_parser_vars.get());
    
    bool is_persistent_variable = false;
    
    Log *log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_EXPRESSIONS));
    
    m_struct_vars->m_struct_laid_out = false;
    
    if (m_struct_members.GetVariable(decl, GetParserID()))
        return true;
    
    ClangExpressionVariableSP var_sp (m_found_entities.GetVariable(decl, GetParserID()));
    
    if (!var_sp)
    {
        var_sp = m_parser_vars->m_persistent_vars->GetVariable(decl, GetParserID());
        is_persistent_variable = true;
    }
    
    if (!var_sp)
        return false;
    
    if (log)
        log->Printf("Adding value for (NamedDecl*)%p [%s - %s] to the structure",
                    decl,
                    name.GetCString(),
                    var_sp->GetName().GetCString());
    
    // We know entity->m_parser_vars is valid because we used a parser variable
    // to find it
    
    ClangExpressionVariable::ParserVars *parser_vars = var_sp->GetParserVars(GetParserID());

    parser_vars->m_llvm_value = value;
    
    if (ClangExpressionVariable::JITVars *jit_vars = var_sp->GetJITVars(GetParserID()))
    {
        // We already laid this out; do not touch
        
        if (log)
            log->Printf("Already placed at 0x%llx", (unsigned long long)jit_vars->m_offset);
    }
    
    var_sp->EnableJITVars(GetParserID());
    
    ClangExpressionVariable::JITVars *jit_vars = var_sp->GetJITVars(GetParserID());

    jit_vars->m_alignment = alignment;
    jit_vars->m_size = size;
    
    m_struct_members.AddVariable(var_sp);
    
    if (m_parser_vars->m_materializer)
    {
        uint32_t offset = 0;

        Error err;

        if (is_persistent_variable)
        {
            offset = m_parser_vars->m_materializer->AddPersistentVariable(var_sp, err);
        }
        else
        {
            if (const lldb_private::Symbol *sym = parser_vars->m_lldb_sym)
                offset = m_parser_vars->m_materializer->AddSymbol(*sym, err);
            else if (const RegisterInfo *reg_info = var_sp->GetRegisterInfo())
                offset = m_parser_vars->m_materializer->AddRegister(*reg_info, err);
            else if (parser_vars->m_lldb_var)
                offset = m_parser_vars->m_materializer->AddVariable(parser_vars->m_lldb_var, err);
        }
        
        if (!err.Success())
            return false;
        
        if (log)
            log->Printf("Placed at 0x%llx", (unsigned long long)offset);
        
        jit_vars->m_offset = offset; // TODO DoStructLayout() should not change this.
    }
    
    return true;
}

bool
ClangExpressionDeclMap::DoStructLayout ()
{
    assert (m_struct_vars.get());
    
    if (m_struct_vars->m_struct_laid_out)
        return true;
    
    if (!m_parser_vars->m_materializer)
        return false;
    
    m_struct_vars->m_struct_alignment = m_parser_vars->m_materializer->GetStructAlignment();
    m_struct_vars->m_struct_size = m_parser_vars->m_materializer->GetStructByteSize();
    m_struct_vars->m_struct_laid_out = true;
    return true;
}

bool ClangExpressionDeclMap::GetStructInfo 
(
    uint32_t &num_elements,
    size_t &size,
    off_t &alignment
)
{
    assert (m_struct_vars.get());
    
    if (!m_struct_vars->m_struct_laid_out)
        return false;
    
    num_elements = m_struct_members.GetSize();
    size = m_struct_vars->m_struct_size;
    alignment = m_struct_vars->m_struct_alignment;
    
    return true;
}

bool 
ClangExpressionDeclMap::GetStructElement 
(
    const NamedDecl *&decl,
    llvm::Value *&value,
    off_t &offset,
    ConstString &name,
    uint32_t index
)
{
    assert (m_struct_vars.get());
    
    if (!m_struct_vars->m_struct_laid_out)
        return false;
    
    if (index >= m_struct_members.GetSize())
        return false;
    
    ClangExpressionVariableSP member_sp(m_struct_members.GetVariableAtIndex(index));
    
    if (!member_sp)
        return false;
    
    ClangExpressionVariable::ParserVars *parser_vars = member_sp->GetParserVars(GetParserID());
    ClangExpressionVariable::JITVars *jit_vars = member_sp->GetJITVars(GetParserID());
    
    if (!parser_vars ||
        !jit_vars ||
        !member_sp->GetValueObject())
        return false;
    
    decl = parser_vars->m_named_decl;
    value = parser_vars->m_llvm_value;
    offset = jit_vars->m_offset;
    name = member_sp->GetName();
        
    return true;
}

bool
ClangExpressionDeclMap::GetFunctionInfo 
(
    const NamedDecl *decl, 
    uint64_t &ptr
)
{
    ClangExpressionVariableSP entity_sp(m_found_entities.GetVariable(decl, GetParserID()));

    if (!entity_sp)
        return false;
    
    // We know m_parser_vars is valid since we searched for the variable by
    // its NamedDecl
    
    ClangExpressionVariable::ParserVars *parser_vars = entity_sp->GetParserVars(GetParserID());

    ptr = parser_vars->m_lldb_value->GetScalar().ULongLong();
    
    return true;
}

static void
FindCodeSymbolInContext
(
    const ConstString &name,
    SymbolContext &sym_ctx,
    SymbolContextList &sc_list
)
{
    SymbolContextList temp_sc_list;
    if (sym_ctx.module_sp)
        sym_ctx.module_sp->FindSymbolsWithNameAndType(name, eSymbolTypeAny, temp_sc_list);
    
    if (!sc_list.GetSize() && sym_ctx.target_sp)
        sym_ctx.target_sp->GetImages().FindSymbolsWithNameAndType(name, eSymbolTypeAny, temp_sc_list);

    unsigned temp_sc_list_size = temp_sc_list.GetSize();
    for (unsigned i = 0; i < temp_sc_list_size; i++)
    {
        SymbolContext sym_ctx;
        temp_sc_list.GetContextAtIndex(i, sym_ctx);
        if (sym_ctx.symbol)
        {
            switch (sym_ctx.symbol->GetType())
            {
                case eSymbolTypeCode:
                case eSymbolTypeResolver:
                    sc_list.Append(sym_ctx);
                    break;

                default:
                    break;
            }
        }
    }
}

bool
ClangExpressionDeclMap::GetFunctionAddress 
(
    const ConstString &name,
    uint64_t &func_addr
)
{
    assert (m_parser_vars.get());
    
    Log *log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_EXPRESSIONS));
    ExecutionContext &exe_ctx = m_parser_vars->m_exe_ctx;
    Target *target = exe_ctx.GetTargetPtr();
    // Back out in all cases where we're not fully initialized
    if (target == NULL)
        return false;
    if (!m_parser_vars->m_sym_ctx.target_sp)
        return false;

    SymbolContextList sc_list;
    
    FindCodeSymbolInContext(name, m_parser_vars->m_sym_ctx, sc_list);

    if (!sc_list.GetSize())
    {
        // We occasionally get debug information in which a const function is reported 
        // as non-const, so the mangled name is wrong.  This is a hack to compensate.
        
        if (!strncmp(name.GetCString(), "_ZN", 3) &&
            strncmp(name.GetCString(), "_ZNK", 4))
        {
            std::string fixed_scratch("_ZNK");
            fixed_scratch.append(name.GetCString() + 3);
            ConstString fixed_name(fixed_scratch.c_str());
            
            if (log)
                log->Printf("Failed to find symbols given non-const name %s; trying %s", name.GetCString(), fixed_name.GetCString());
            
            FindCodeSymbolInContext(fixed_name, m_parser_vars->m_sym_ctx, sc_list);
        }
    }
    
    if (!sc_list.GetSize())
        return false;

    SymbolContext sym_ctx;
    sc_list.GetContextAtIndex(0, sym_ctx);

    const Address *func_so_addr = NULL;
    bool is_indirect_function = false;

    if (sym_ctx.function)
        func_so_addr = &sym_ctx.function->GetAddressRange().GetBaseAddress();
    else if (sym_ctx.symbol) {
        func_so_addr = &sym_ctx.symbol->GetAddress();
        is_indirect_function = sym_ctx.symbol->IsIndirect();
    } else
        return false;

    if (!func_so_addr || !func_so_addr->IsValid())
        return false;

    func_addr = func_so_addr->GetCallableLoadAddress (target, is_indirect_function);

    return true;
}

addr_t
ClangExpressionDeclMap::GetSymbolAddress (Target &target, Process *process, const ConstString &name, lldb::SymbolType symbol_type)
{
    SymbolContextList sc_list;
    
    target.GetImages().FindSymbolsWithNameAndType(name, symbol_type, sc_list);
    
    const uint32_t num_matches = sc_list.GetSize();
    addr_t symbol_load_addr = LLDB_INVALID_ADDRESS;

    for (uint32_t i=0; i<num_matches && (symbol_load_addr == 0 || symbol_load_addr == LLDB_INVALID_ADDRESS); i++)
    {
        SymbolContext sym_ctx;
        sc_list.GetContextAtIndex(i, sym_ctx);
    
        const Address *sym_address = &sym_ctx.symbol->GetAddress();
        
        if (!sym_address || !sym_address->IsValid())
            return LLDB_INVALID_ADDRESS;
        
        if (sym_address)
        {
            switch (sym_ctx.symbol->GetType())
            {
                case eSymbolTypeCode:
                case eSymbolTypeTrampoline:
                    symbol_load_addr = sym_address->GetCallableLoadAddress (&target);
                    break;

                case eSymbolTypeResolver:
                    symbol_load_addr = sym_address->GetCallableLoadAddress (&target, true);
                    break;

                case eSymbolTypeData:
                case eSymbolTypeRuntime:
                case eSymbolTypeVariable:
                case eSymbolTypeLocal:
                case eSymbolTypeParam:
                case eSymbolTypeInvalid:
                case eSymbolTypeAbsolute:
                case eSymbolTypeException:
                case eSymbolTypeSourceFile:
                case eSymbolTypeHeaderFile:
                case eSymbolTypeObjectFile:
                case eSymbolTypeCommonBlock:
                case eSymbolTypeBlock:
                case eSymbolTypeVariableType:
                case eSymbolTypeLineEntry:
                case eSymbolTypeLineHeader:
                case eSymbolTypeScopeBegin:
                case eSymbolTypeScopeEnd:
                case eSymbolTypeAdditional:
                case eSymbolTypeCompiler:
                case eSymbolTypeInstrumentation:
                case eSymbolTypeUndefined:
                case eSymbolTypeObjCClass:
                case eSymbolTypeObjCMetaClass:
                case eSymbolTypeObjCIVar:
                    symbol_load_addr = sym_address->GetLoadAddress (&target);
                    break;
            }
        }
    }
    
    if (symbol_load_addr == LLDB_INVALID_ADDRESS && process)
    {
        ObjCLanguageRuntime *runtime = process->GetObjCLanguageRuntime();
        
        if (runtime)
        {
            symbol_load_addr = runtime->LookupRuntimeSymbol(name);
        }
    }
    
    return symbol_load_addr;
}

addr_t
ClangExpressionDeclMap::GetSymbolAddress (const ConstString &name, lldb::SymbolType symbol_type)
{
    assert (m_parser_vars.get());
    
    if (!m_parser_vars->m_exe_ctx.GetTargetPtr())
        return false;
    
    return GetSymbolAddress(m_parser_vars->m_exe_ctx.GetTargetRef(), m_parser_vars->m_exe_ctx.GetProcessPtr(), name, symbol_type);
}

// Interface for IRInterpreter

Value 
ClangExpressionDeclMap::WrapBareAddress (lldb::addr_t addr)
{
    Value ret;

    ret.SetContext(Value::eContextTypeInvalid, NULL);

    if (m_parser_vars->m_exe_ctx.GetProcessPtr())
        ret.SetValueType(Value::eValueTypeLoadAddress);
    else
        ret.SetValueType(Value::eValueTypeFileAddress);

    ret.GetScalar() = (unsigned long long)addr;

    return ret;
}

bool
ClangExpressionDeclMap::WriteTarget (lldb_private::IRMemoryMap &map,
                                     lldb_private::Value &value,
                                     const uint8_t *data,
                                     size_t length)
{
    assert (m_parser_vars.get());
    
    ExecutionContext &exe_ctx = m_parser_vars->m_exe_ctx;
    
    Process *process = exe_ctx.GetProcessPtr();
    if (value.GetContextType() == Value::eContextTypeRegisterInfo)
    {
        if (!process)
            return false;
        
        RegisterContext *reg_ctx = exe_ctx.GetRegisterContext();
        RegisterInfo *reg_info = value.GetRegisterInfo();
        
        if (!reg_ctx)
            return false;
        
        lldb_private::RegisterValue reg_value;
        Error err;
        
        if (!reg_value.SetFromMemoryData (reg_info, data, length, process->GetByteOrder(), err))
            return false;
        
        return reg_ctx->WriteRegister(reg_info, reg_value);
    }
    else
    {
        switch (value.GetValueType())
        {
        default:
            return false;
        case Value::eValueTypeFileAddress:
            {
                if (!process)
                    return false;
                
                Target *target = exe_ctx.GetTargetPtr();
                Address file_addr;
                
                if (!target->GetImages().ResolveFileAddress((lldb::addr_t)value.GetScalar().ULongLong(), file_addr))
                    return false;
                
                lldb::addr_t load_addr = file_addr.GetLoadAddress(target);
                
                Error err;
                map.WriteMemory(load_addr, data, length, err);
                
                return err.Success();
            }
        case Value::eValueTypeLoadAddress:
            {
                if (!process)
                    return false;
                
                Error err;
                map.WriteMemory((lldb::addr_t)value.GetScalar().ULongLong(), data, length, err);
    
                return err.Success();
            }
        case Value::eValueTypeHostAddress:
            {
                if (value.GetScalar().ULongLong() == 0 || data == NULL)
                    return false;
                memcpy ((void *)value.GetScalar().ULongLong(), data, length);
                return true;
            }
        case Value::eValueTypeScalar:
            return false;
        }
    }
}

bool
ClangExpressionDeclMap::ReadTarget (IRMemoryMap &map,
                                    uint8_t *data,
                                    lldb_private::Value &value,
                                    size_t length)
{
    assert (m_parser_vars.get());
    
    ExecutionContext &exe_ctx = m_parser_vars->m_exe_ctx;

    Process *process = exe_ctx.GetProcessPtr();

    if (value.GetContextType() == Value::eContextTypeRegisterInfo)
    {
        if (!process)
            return false;
        
        RegisterContext *reg_ctx = exe_ctx.GetRegisterContext();
        RegisterInfo *reg_info = value.GetRegisterInfo();
        
        if (!reg_ctx)
            return false;
        
        lldb_private::RegisterValue reg_value;
        Error err;
        
        if (!reg_ctx->ReadRegister(reg_info, reg_value))
            return false;
        
        return reg_value.GetAsMemoryData(reg_info, data, length, process->GetByteOrder(), err);        
    }
    else
    {
        switch (value.GetValueType())
        {
            default:
                return false;
            case Value::eValueTypeFileAddress:
            {
                Target *target = exe_ctx.GetTargetPtr();
                if (target == NULL)
                    return false;
                
                Address file_addr;
                
                if (!target->GetImages().ResolveFileAddress((lldb::addr_t)value.GetScalar().ULongLong(), file_addr))
                    return false;
                
                Error err;
                target->ReadMemory(file_addr, false, data, length, err);
                
                return err.Success();
            }
            case Value::eValueTypeLoadAddress:
            {
                Error err;
                map.ReadMemory(data, (lldb::addr_t)value.GetScalar().ULongLong(), length, err);
                
                return err.Success();
            }
            case Value::eValueTypeHostAddress:
            {
                void *host_addr = (void*)value.GetScalar().ULongLong();
                
                if (!host_addr)
                    return false;
                
                memcpy (data, host_addr, length);
                return true;
            }
            case Value::eValueTypeScalar:
                return false;
        }
    }
}

lldb_private::Value
ClangExpressionDeclMap::LookupDecl (clang::NamedDecl *decl, ClangExpressionVariable::FlagType &flags)
{
    assert (m_parser_vars.get());
            
    ClangExpressionVariableSP expr_var_sp (m_found_entities.GetVariable(decl, GetParserID()));
    ClangExpressionVariableSP persistent_var_sp (m_parser_vars->m_persistent_vars->GetVariable(decl, GetParserID()));
    
    if (isa<FunctionDecl>(decl))
    {
        ClangExpressionVariableSP entity_sp(m_found_entities.GetVariable(decl, GetParserID()));
        
        if (!entity_sp)
            return Value();
        
        // We know m_parser_vars is valid since we searched for the variable by
        // its NamedDecl
        
        ClangExpressionVariable::ParserVars *parser_vars = entity_sp->GetParserVars(GetParserID());
        
        return *parser_vars->m_lldb_value;
    }
    
    if (expr_var_sp)
    {
        flags = expr_var_sp->m_flags;

        ClangExpressionVariable::ParserVars *parser_vars = expr_var_sp->GetParserVars(GetParserID());

        if (!parser_vars)
            return Value();
        
        bool is_reference = expr_var_sp->m_flags & ClangExpressionVariable::EVTypeIsReference;

        if (parser_vars->m_lldb_var)
        {
            std::unique_ptr<Value> value(GetVariableValue(parser_vars->m_lldb_var, NULL));
            
            if (is_reference && value.get() && value->GetValueType() == Value::eValueTypeLoadAddress)
            {
                Process *process = m_parser_vars->m_exe_ctx.GetProcessPtr();
                
                if (!process)
                    return Value();
                
                lldb::addr_t value_addr = value->GetScalar().ULongLong();
                Error read_error;
                addr_t ref_value = process->ReadPointerFromMemory (value_addr, read_error);
                
                if (!read_error.Success())
                    return Value();
                
                value->GetScalar() = (unsigned long long)ref_value;
            }
        
            if (value.get())
                return *value;
            else
                return Value();
        }
        else if (parser_vars->m_lldb_sym)
        {
            const Address sym_address = parser_vars->m_lldb_sym->GetAddress();
            
            if (!sym_address.IsValid())
                return Value();
                        
            Value ret;
        
            ProcessSP process_sp (m_parser_vars->m_exe_ctx.GetProcessSP());
            
            if (process_sp)
            {
                uint64_t symbol_load_addr = sym_address.GetLoadAddress(&process_sp->GetTarget());
                
                ret.GetScalar() = symbol_load_addr;
                ret.SetValueType(Value::eValueTypeLoadAddress);
            }
            else 
            {
                uint64_t symbol_file_addr = sym_address.GetFileAddress();
                
                ret.GetScalar() = symbol_file_addr;
                ret.SetValueType(Value::eValueTypeFileAddress);
            }
            
            return ret;
        }
        else if (RegisterInfo *reg_info = expr_var_sp->GetRegisterInfo())
        {
            StackFrame *frame = m_parser_vars->m_exe_ctx.GetFramePtr();
            
            if (!frame)
                return Value();
            
            RegisterContextSP reg_context_sp(frame->GetRegisterContextSP());
            
            RegisterValue reg_value;
            
            if (!reg_context_sp->ReadRegister(reg_info, reg_value))
                return Value();
            
            Value ret;
            
            ret.SetContext(Value::eContextTypeRegisterInfo, reg_info);
            if (reg_info->encoding == eEncodingVector) 
			{
                if (ret.SetVectorBytes((uint8_t *)reg_value.GetBytes(), reg_value.GetByteSize(), reg_value.GetByteOrder()))
                    ret.SetScalarFromVector();
            }
            else if (!reg_value.GetScalarValue(ret.GetScalar()))
				return Value();
            
            return ret;
        }
        else
        {
            return Value();
        }
    }
    else if (persistent_var_sp)
    {
        flags = persistent_var_sp->m_flags;
        
        if ((persistent_var_sp->m_flags & ClangExpressionVariable::EVIsProgramReference ||
             persistent_var_sp->m_flags & ClangExpressionVariable::EVIsLLDBAllocated) &&
            persistent_var_sp->m_live_sp &&
            ((persistent_var_sp->m_live_sp->GetValue().GetValueType() == Value::eValueTypeLoadAddress &&
              m_parser_vars->m_exe_ctx.GetProcessSP() &&
              m_parser_vars->m_exe_ctx.GetProcessSP()->IsAlive()) ||
             (persistent_var_sp->m_live_sp->GetValue().GetValueType() == Value::eValueTypeFileAddress)))
        {
            return persistent_var_sp->m_live_sp->GetValue();
        }
        else
        {
            lldb_private::Value ret;
            ret.SetValueType(Value::eValueTypeHostAddress);
            ret.SetContext(Value::eContextTypeInvalid, NULL);
            ret.GetScalar() = (lldb::addr_t)persistent_var_sp->GetValueBytes();
            return ret;
        }
    }
    else
    {
        return Value();
    }
}

Value
ClangExpressionDeclMap::GetSpecialValue (const ConstString &name)
{
    assert(m_parser_vars.get());
    
    StackFrame *frame = m_parser_vars->m_exe_ctx.GetFramePtr();
    
    if (!frame)
        return Value();
    
    VariableList *vars = frame->GetVariableList(false);
    
    if (!vars)
        return Value();
    
    lldb::VariableSP var = vars->FindVariable(name);
    
    if (!var ||
        !var->IsInScope(frame) || 
        !var->LocationIsValidForFrame (frame))
        return Value();
    
    std::unique_ptr<Value> value(GetVariableValue(var, NULL));
    
    if (value.get() && value->GetValueType() == Value::eValueTypeLoadAddress)
    {
        Process *process = m_parser_vars->m_exe_ctx.GetProcessPtr();
        
        if (!process)
            return Value();
        
        lldb::addr_t value_addr = value->GetScalar().ULongLong();
        Error read_error;
        addr_t ptr_value = process->ReadPointerFromMemory (value_addr, read_error);
        
        if (!read_error.Success())
            return Value();
        
        value->GetScalar() = (unsigned long long)ptr_value;
    }
    
    if (value.get())
        return *value;
    else
        return Value();
}

// Interface for CommandObjectExpression

bool 
ClangExpressionDeclMap::Materialize 
(
    IRMemoryMap &map,
    lldb::addr_t &struct_address,
    Error &err
)
{
    if (!m_parser_vars.get())
        return false;
    
    EnableMaterialVars();
    
    m_material_vars->m_process = m_parser_vars->m_exe_ctx.GetProcessPtr();
    
    bool result = DoMaterialize(false /* dematerialize */,
                                map,
                                LLDB_INVALID_ADDRESS /* top of stack frame */, 
                                LLDB_INVALID_ADDRESS /* bottom of stack frame */, 
                                NULL, /* result SP */
                                err);
    
    if (result)
        struct_address = m_material_vars->m_materialized_location;
    
    return result;
}

bool 
ClangExpressionDeclMap::GetObjectPointer
(
    lldb::addr_t &object_ptr,
    ConstString &object_name,
    Error &err,
    bool suppress_type_check
)
{
    assert (m_struct_vars.get());
    
    Target *target = m_parser_vars->m_exe_ctx.GetTargetPtr();
    Process *process = m_parser_vars->m_exe_ctx.GetProcessPtr();
    StackFrame *frame = m_parser_vars->m_exe_ctx.GetFramePtr();

    if (frame == NULL || process == NULL || target == NULL)
    {
        err.SetErrorStringWithFormat("Couldn't load '%s' because the context is incomplete", object_name.AsCString());
        return false;
    }
    
    if (!m_struct_vars->m_object_pointer_type.GetOpaqueQualType())
    {
        err.SetErrorStringWithFormat("Couldn't load '%s' because its type is unknown", object_name.AsCString());
        return false;
    }
    
    const bool object_pointer = true;
    
    VariableSP object_ptr_var = FindVariableInScope (*frame,
                                                     object_name, 
                                                     (suppress_type_check ? NULL : &m_struct_vars->m_object_pointer_type),
                                                     object_pointer);
    
    if (!object_ptr_var)
    {
        err.SetErrorStringWithFormat("Couldn't find '%s' with appropriate type in scope", object_name.AsCString());
        return false;
    }
    
    std::unique_ptr<lldb_private::Value> location_value(GetVariableValue(object_ptr_var,
                                                                       NULL));
    
    if (!location_value.get())
    {
        err.SetErrorStringWithFormat("Couldn't get the location for '%s'", object_name.GetCString());
        return false;
    }
    
    switch (location_value->GetValueType())
    {
    default:
        err.SetErrorStringWithFormat("'%s' is not in memory; LLDB must be extended to handle registers", object_name.GetCString());
        return false;
    case Value::eValueTypeLoadAddress:
        {
            lldb::addr_t value_addr = location_value->GetScalar().ULongLong();
            uint32_t address_byte_size = target->GetArchitecture().GetAddressByteSize();
            
            if (ClangASTType::GetClangTypeBitWidth(m_struct_vars->m_object_pointer_type.GetASTContext(), 
                                                   m_struct_vars->m_object_pointer_type.GetOpaqueQualType()) != address_byte_size * 8)
            {
                err.SetErrorStringWithFormat("'%s' is not of an expected pointer size", object_name.GetCString());
                return false;
            }
            
            Error read_error;
            object_ptr = process->ReadPointerFromMemory (value_addr, read_error);
            if (read_error.Fail() || object_ptr == LLDB_INVALID_ADDRESS)
            {
                err.SetErrorStringWithFormat("Coldn't read '%s' from the target: %s", object_name.GetCString(), read_error.AsCString());
                return false;
            }            
            return true;
        }
    case Value::eValueTypeScalar:
        {
            if (location_value->GetContextType() != Value::eContextTypeRegisterInfo)
            {
                StreamString ss;
                location_value->Dump(&ss);
                
                err.SetErrorStringWithFormat("%s is a scalar of unhandled type: %s", object_name.GetCString(), ss.GetString().c_str());
                return false;
            }
                        
            RegisterInfo *reg_info = location_value->GetRegisterInfo();
            
            if (!reg_info)
            {
                err.SetErrorStringWithFormat("Couldn't get the register information for %s", object_name.GetCString());
                return false;
            }
            
            RegisterContext *reg_ctx = m_parser_vars->m_exe_ctx.GetRegisterContext();
            
            if (!reg_ctx)
            {
                err.SetErrorStringWithFormat("Couldn't read register context to read %s from %s", object_name.GetCString(), reg_info->name);
                return false;
            }
            
            uint32_t register_number = reg_info->kinds[lldb::eRegisterKindLLDB];
            
            object_ptr = reg_ctx->ReadRegisterAsUnsigned(register_number, 0x0);
            
            return true;
        }
    }
}

bool 
ClangExpressionDeclMap::Dematerialize 
(
    ClangExpressionVariableSP &result_sp,
    IRMemoryMap &map,
    lldb::addr_t stack_frame_top,
    lldb::addr_t stack_frame_bottom,
    Error &err
)
{
    return DoMaterialize(true, map, stack_frame_top, stack_frame_bottom, &result_sp, err);
    
    DidDematerialize();
}

void
ClangExpressionDeclMap::DidDematerialize()
{
    if (m_material_vars.get())
    {
        if (m_material_vars->m_materialized_location)
        {        
            //#define SINGLE_STEP_EXPRESSIONS
            
#ifndef SINGLE_STEP_EXPRESSIONS
            m_material_vars->m_process->DeallocateMemory(m_material_vars->m_materialized_location);
#endif
            m_material_vars->m_materialized_location = 0;
        }
        
        DisableMaterialVars();
    }
}

bool
ClangExpressionDeclMap::DumpMaterializedStruct
(
    Stream &s,
    Error &err
)
{
    assert (m_struct_vars.get());
    assert (m_material_vars.get());
    
    if (!m_struct_vars->m_struct_laid_out)
    {
        err.SetErrorString("Structure hasn't been laid out yet");
        return false;
    }
    Process *process = m_parser_vars->m_exe_ctx.GetProcessPtr();

    if (!process)
    {
        err.SetErrorString("Couldn't find the process");
        return false;
    }
    
    Target *target = m_parser_vars->m_exe_ctx.GetTargetPtr();
    if (!target)
    {
        err.SetErrorString("Couldn't find the target");
        return false;
    }
    
    if (!m_material_vars->m_materialized_location)
    {
        err.SetErrorString("No materialized location");
        return false;
    }
    
    lldb::DataBufferSP data_sp(new DataBufferHeap(m_struct_vars->m_struct_size, 0));    
    
    Error error;
    if (process->ReadMemory (m_material_vars->m_materialized_location, 
                                     data_sp->GetBytes(), 
                                     data_sp->GetByteSize(), error) != data_sp->GetByteSize())
    {
        err.SetErrorStringWithFormat ("Couldn't read struct from the target: %s", error.AsCString());
        return false;
    }
    
    DataExtractor extractor(data_sp, process->GetByteOrder(), target->GetArchitecture().GetAddressByteSize());
    
    for (size_t member_idx = 0, num_members = m_struct_members.GetSize();
         member_idx < num_members;
         ++member_idx)
    {
        ClangExpressionVariableSP member_sp(m_struct_members.GetVariableAtIndex(member_idx));
        
        if (!member_sp)
            return false;

        s.Printf("[%s]\n", member_sp->GetName().GetCString());
        
        ClangExpressionVariable::JITVars *jit_vars = member_sp->GetJITVars(GetParserID());

        if (!jit_vars)
            return false;
        
        extractor.Dump (&s,                                                                          // stream
                        jit_vars->m_offset,                                                          // offset
                        lldb::eFormatBytesWithASCII,                                                 // format
                        1,                                                                           // byte size of individual entries
                        jit_vars->m_size,                                                            // number of entries
                        16,                                                                          // entries per line
                        m_material_vars->m_materialized_location + jit_vars->m_offset,               // address to print
                        0,                                                                           // bit size (bitfields only; 0 means ignore)
                        0);                                                                          // bit alignment (bitfields only; 0 means ignore)
        
        s.PutChar('\n');
    }
    
    return true;
}

bool 
ClangExpressionDeclMap::DoMaterialize 
(
    bool dematerialize,
    IRMemoryMap &map,
    lldb::addr_t stack_frame_top,
    lldb::addr_t stack_frame_bottom,
    lldb::ClangExpressionVariableSP *result_sp_ptr,
    Error &err
)
{
    if (result_sp_ptr)
        result_sp_ptr->reset();

    assert (m_struct_vars.get());
    
    Log *log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_EXPRESSIONS));
    
    if (!m_struct_vars->m_struct_laid_out)
    {
        err.SetErrorString("Structure hasn't been laid out yet");
        return false;
    }
    
    StackFrame *frame = m_parser_vars->m_exe_ctx.GetFramePtr();
    if (!frame)
    {
        err.SetErrorString("Received null execution frame");
        return false;
    }
    Target *target = m_parser_vars->m_exe_ctx.GetTargetPtr();
    
    ClangPersistentVariables &persistent_vars = target->GetPersistentVariables();
        
    if (!m_struct_vars->m_struct_size)
    {
        if (log)
            log->PutCString("Not bothering to allocate a struct because no arguments are needed");
        
        m_material_vars->m_allocated_area = 0UL;
        
        return true;
    }
    
    if (!m_parser_vars->m_materializer)
    {
        err.SetErrorString("No materializer");
        
        return false;
    }
    
    lldb::StackFrameSP frame_sp = frame->shared_from_this();
    ClangExpressionVariableSP result_sp;
    
    if (dematerialize)
    {
        Error dematerialize_error;
        
        bool ret = true;
        
        ClangExpressionVariableSP result;
        
        m_material_vars->m_dematerializer_sp->Dematerialize(dematerialize_error, result, stack_frame_top, stack_frame_bottom);
        m_material_vars->m_dematerializer_sp.reset();
        
        if (!dematerialize_error.Success())
        {
            err.SetErrorStringWithFormat("Couldn't dematerialize: %s", dematerialize_error.AsCString());
            ret = false;
        }
        else
        {
            Error free_error;
            map.Free(m_material_vars->m_materialized_location, free_error);
            if (!free_error.Success())
            {
                err.SetErrorStringWithFormat("Couldn't free struct from materialization: %s", free_error.AsCString());
                ret = false;
            }
        }
        
        if (ret)
        {
            for (uint64_t member_index = 0, num_members = m_struct_members.GetSize();
                 member_index < num_members;
                 ++member_index)
            {
                ClangExpressionVariableSP member_sp(m_struct_members.GetVariableAtIndex(member_index));

                if (!m_found_entities.ContainsVariable (member_sp))
                {
                    // No need to look for presistent variables if the name doesn't start
                    // with with a '$' character...
                    if (member_sp->GetName().AsCString ("!")[0] == '$' && persistent_vars.ContainsVariable(member_sp))
                    {
                        if (member_sp->GetName() == m_struct_vars->m_result_name)
                        {
                            if (log)
                                log->PutCString("Found result member in the struct");
                            
                            if (result_sp_ptr)
                                *result_sp_ptr = member_sp;
                            
                            break;
                        }
                    }
                }
            }
        }
        
        return ret;
    }
    else
    {
        Error malloc_error;
        
        m_material_vars->m_allocated_area = LLDB_INVALID_ADDRESS;
        m_material_vars->m_materialized_location = map.Malloc(m_parser_vars->m_materializer->GetStructByteSize(),
                                                              m_parser_vars->m_materializer->GetStructAlignment(),
                                                              lldb::ePermissionsReadable | lldb::ePermissionsWritable,
                                                              IRMemoryMap::eAllocationPolicyMirror,
                                                              malloc_error);
        
        if (!malloc_error.Success())
        {
            err.SetErrorStringWithFormat("Couldn't malloc struct for materialization: %s", malloc_error.AsCString());
            
            return false;
        }
        
        Error materialize_error;
        
        m_material_vars->m_dematerializer_sp = m_parser_vars->m_materializer->Materialize(frame_sp, map, m_material_vars->m_materialized_location, materialize_error);
        
        if (!materialize_error.Success())
        {
            err.SetErrorStringWithFormat("Couldn't materialize: %s", materialize_error.AsCString());
            
            return false;
        }
        
        return true;
    }
}

lldb::VariableSP
ClangExpressionDeclMap::FindVariableInScope
(
    StackFrame &frame,
    const ConstString &name,
    TypeFromUser *type,
    bool object_pointer
)
{    
    Log *log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_EXPRESSIONS));
    
    ValueObjectSP valobj;
    VariableSP var_sp;
    Error err;
    
    valobj = frame.GetValueForVariableExpressionPath(name.GetCString(),
                                                     eNoDynamicValues, 
                                                     StackFrame::eExpressionPathOptionCheckPtrVsMember ||
                                                     StackFrame::eExpressionPathOptionsAllowDirectIVarAccess ||
                                                     StackFrame::eExpressionPathOptionsNoFragileObjcIvar ||
                                                     StackFrame::eExpressionPathOptionsNoSyntheticChildren ||
                                                     StackFrame::eExpressionPathOptionsNoSyntheticArrayRange,
                                                     var_sp,
                                                     err);
        
    if (!err.Success() ||
        !var_sp ||
        !var_sp->IsInScope(&frame) ||
        !var_sp->LocationIsValidForFrame (&frame))
        return lldb::VariableSP();

    if (var_sp)
    {
        if (!type)
            return var_sp;
        
        TypeFromUser candidate_type(var_sp->GetType()->GetClangFullType(),
                                    var_sp->GetType()->GetClangAST());
        
        if (candidate_type.GetASTContext() != type->GetASTContext())
        {
            if (log)
                log->PutCString("Skipping a candidate variable because of different AST contexts");
            return lldb::VariableSP();
        }
        
        if (object_pointer)
        {
            clang::QualType desired_qual_type = clang::QualType::getFromOpaquePtr(type->GetOpaqueQualType());
            clang::QualType candidate_qual_type = clang::QualType::getFromOpaquePtr(candidate_type.GetOpaqueQualType());
            
            const clang::ObjCObjectPointerType *desired_objc_ptr_type = desired_qual_type->getAs<clang::ObjCObjectPointerType>();
            const clang::ObjCObjectPointerType *candidate_objc_ptr_type = desired_qual_type->getAs<clang::ObjCObjectPointerType>();
            
            if (desired_objc_ptr_type && candidate_objc_ptr_type) {
                clang::QualType desired_target_type = desired_objc_ptr_type->getPointeeType().getUnqualifiedType();
                clang::QualType candidate_target_type = candidate_objc_ptr_type->getPointeeType().getUnqualifiedType();
                
                if (ClangASTContext::AreTypesSame(type->GetASTContext(),
                                                  desired_target_type.getAsOpaquePtr(),
                                                  candidate_target_type.getAsOpaquePtr()))
                    return var_sp;
            }
            
            const clang::PointerType *desired_ptr_type = desired_qual_type->getAs<clang::PointerType>();
            const clang::PointerType *candidate_ptr_type = candidate_qual_type->getAs<clang::PointerType>();
            
            if (desired_ptr_type && candidate_ptr_type) {
                clang::QualType desired_target_type = desired_ptr_type->getPointeeType().getUnqualifiedType();
                clang::QualType candidate_target_type = candidate_ptr_type->getPointeeType().getUnqualifiedType();
                
                if (ClangASTContext::AreTypesSame(type->GetASTContext(),
                                                  desired_target_type.getAsOpaquePtr(),
                                                  candidate_target_type.getAsOpaquePtr()))
                    return var_sp;
            }
            
            return lldb::VariableSP();
        }
        else
        {
            if (ClangASTContext::AreTypesSame(type->GetASTContext(),
                                               type->GetOpaqueQualType(), 
                                               var_sp->GetType()->GetClangFullType()))
                return var_sp;
        }
    }

    return lldb::VariableSP();
}

const Symbol *
ClangExpressionDeclMap::FindGlobalDataSymbol (Target &target,
                                              const ConstString &name)
{
    SymbolContextList sc_list;
    
    target.GetImages().FindSymbolsWithNameAndType(name, eSymbolTypeAny, sc_list);
    
    const uint32_t matches = sc_list.GetSize();
    for (uint32_t i=0; i<matches; ++i)
    {
        SymbolContext sym_ctx;
        sc_list.GetContextAtIndex(i, sym_ctx);
        if (sym_ctx.symbol)
        {
            const Symbol *symbol = sym_ctx.symbol;
            const Address *sym_address = &symbol->GetAddress();
            
            if (sym_address && sym_address->IsValid())
            {
                switch (symbol->GetType())
                {
                    case eSymbolTypeData:
                    case eSymbolTypeRuntime:
                    case eSymbolTypeAbsolute:
                    case eSymbolTypeObjCClass:
                    case eSymbolTypeObjCMetaClass:
                    case eSymbolTypeObjCIVar:
                        if (symbol->GetDemangledNameIsSynthesized())
                        {
                            // If the demangled name was synthesized, then don't use it
                            // for expressions. Only let the symbol match if the mangled
                            // named matches for these symbols.
                            if (symbol->GetMangled().GetMangledName() != name)
                                break;
                        }
                        return symbol;

                    case eSymbolTypeCode: // We already lookup functions elsewhere
                    case eSymbolTypeVariable:
                    case eSymbolTypeLocal:
                    case eSymbolTypeParam:
                    case eSymbolTypeTrampoline:
                    case eSymbolTypeInvalid:
                    case eSymbolTypeException:
                    case eSymbolTypeSourceFile:
                    case eSymbolTypeHeaderFile:
                    case eSymbolTypeObjectFile:
                    case eSymbolTypeCommonBlock:
                    case eSymbolTypeBlock:
                    case eSymbolTypeVariableType:
                    case eSymbolTypeLineEntry:
                    case eSymbolTypeLineHeader:
                    case eSymbolTypeScopeBegin:
                    case eSymbolTypeScopeEnd:
                    case eSymbolTypeAdditional:
                    case eSymbolTypeCompiler:
                    case eSymbolTypeInstrumentation:
                    case eSymbolTypeUndefined:
                    case eSymbolTypeResolver:
                        break;
                }
            }
        }
    }
    
    return NULL;
}

lldb::VariableSP
ClangExpressionDeclMap::FindGlobalVariable
(
    Target &target,
    ModuleSP &module,
    const ConstString &name,
    ClangNamespaceDecl *namespace_decl,
    TypeFromUser *type
)
{
    VariableList vars;
    
    if (module && namespace_decl)
        module->FindGlobalVariables (name, namespace_decl, true, -1, vars);
    else
        target.GetImages().FindGlobalVariables(name, true, -1, vars);
    
    if (vars.GetSize())
    {
        if (type)
        {
            for (size_t i = 0; i < vars.GetSize(); ++i)
            {
                VariableSP var_sp = vars.GetVariableAtIndex(i);
                
                if (type->GetASTContext() == var_sp->GetType()->GetClangAST())
                {
                    if (ClangASTContext::AreTypesSame(type->GetASTContext(), type->GetOpaqueQualType(), var_sp->GetType()->GetClangFullType()))
                        return var_sp;
                }
            }
        }
        else
        {
            return vars.GetVariableAtIndex(0);
        }
    }
    
    return VariableSP();
}

// Interface for ClangASTSource

void
ClangExpressionDeclMap::FindExternalVisibleDecls (NameSearchContext &context)
{
    assert (m_ast_context);
    
    ClangASTMetrics::RegisterVisibleQuery();
    
    const ConstString name(context.m_decl_name.getAsString().c_str());
    
    Log *log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_EXPRESSIONS));
    
    if (GetImportInProgress())
    {
        if (log && log->GetVerbose())
            log->Printf("Ignoring a query during an import");
        return;
    }
    
    static unsigned int invocation_id = 0;
    unsigned int current_id = invocation_id++;
    
    if (log)
    {
        if (!context.m_decl_context)
            log->Printf("ClangExpressionDeclMap::FindExternalVisibleDecls[%u] for '%s' in a NULL DeclContext", current_id, name.GetCString());
        else if (const NamedDecl *context_named_decl = dyn_cast<NamedDecl>(context.m_decl_context))
            log->Printf("ClangExpressionDeclMap::FindExternalVisibleDecls[%u] for '%s' in '%s'", current_id, name.GetCString(), context_named_decl->getNameAsString().c_str());
        else
            log->Printf("ClangExpressionDeclMap::FindExternalVisibleDecls[%u] for '%s' in a '%s'", current_id, name.GetCString(), context.m_decl_context->getDeclKindName());
    }
            
    if (const NamespaceDecl *namespace_context = dyn_cast<NamespaceDecl>(context.m_decl_context))
    {
        ClangASTImporter::NamespaceMapSP namespace_map = m_ast_importer->GetNamespaceMap(namespace_context);
        
        if (log && log->GetVerbose())
            log->Printf("  CEDM::FEVD[%u] Inspecting (NamespaceMap*)%p (%d entries)", 
                        current_id, 
                        namespace_map.get(), 
                        (int)namespace_map->size());
        
        if (!namespace_map)
            return;
        
        for (ClangASTImporter::NamespaceMap::iterator i = namespace_map->begin(), e = namespace_map->end();
             i != e;
             ++i)
        {
            if (log)
                log->Printf("  CEDM::FEVD[%u] Searching namespace %s in module %s",
                            current_id,
                            i->second.GetNamespaceDecl()->getNameAsString().c_str(),
                            i->first->GetFileSpec().GetFilename().GetCString());
                
            FindExternalVisibleDecls(context,
                                     i->first,
                                     i->second,
                                     current_id);
        }
    }
    else if (isa<TranslationUnitDecl>(context.m_decl_context))
    {
        ClangNamespaceDecl namespace_decl;
        
        if (log)
            log->Printf("  CEDM::FEVD[%u] Searching the root namespace", current_id);
        
        FindExternalVisibleDecls(context,
                                 lldb::ModuleSP(),
                                 namespace_decl,
                                 current_id);
    }
    
    if (!context.m_found.variable)
        ClangASTSource::FindExternalVisibleDecls(context);
}

void 
ClangExpressionDeclMap::FindExternalVisibleDecls (NameSearchContext &context, 
                                                  lldb::ModuleSP module_sp,
                                                  ClangNamespaceDecl &namespace_decl,
                                                  unsigned int current_id)
{
    assert (m_ast_context);
    
    Log *log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_EXPRESSIONS));
    
    SymbolContextList sc_list;
    
    const ConstString name(context.m_decl_name.getAsString().c_str());
    
    const char *name_unique_cstr = name.GetCString();
    
    if (name_unique_cstr == NULL)
        return;
    
    static ConstString id_name("id");
    static ConstString Class_name("Class");
    
    if (name == id_name || name == Class_name)
        return;
    
    // Only look for functions by name out in our symbols if the function 
    // doesn't start with our phony prefix of '$'
    Target *target = m_parser_vars->m_exe_ctx.GetTargetPtr();
    StackFrame *frame = m_parser_vars->m_exe_ctx.GetFramePtr();
    if (name_unique_cstr[0] == '$' && !namespace_decl)
    {
        static ConstString g_lldb_class_name ("$__lldb_class");
        
        if (name == g_lldb_class_name)
        {
            // Clang is looking for the type of "this"
                        
            if (frame == NULL)
                return;
            
            SymbolContext sym_ctx = frame->GetSymbolContext(lldb::eSymbolContextFunction);
            
            if (!sym_ctx.function)
                return;
            
            // Get the block that defines the function
            Block *function_block = sym_ctx.GetFunctionBlock();

            if (!function_block)
                return;

            clang::DeclContext *decl_context = function_block->GetClangDeclContext();
            
            if (!decl_context)
                return;
            
            clang::CXXMethodDecl *method_decl = llvm::dyn_cast<clang::CXXMethodDecl>(decl_context);
            
            if (method_decl)
            {
                clang::CXXRecordDecl *class_decl = method_decl->getParent();
                
                QualType class_qual_type(class_decl->getTypeForDecl(), 0);
                
                TypeFromUser class_user_type (class_qual_type.getAsOpaquePtr(),
                                              &class_decl->getASTContext());
                
                if (log)
                {
                    ASTDumper ast_dumper(class_qual_type);
                    log->Printf("  CEDM::FEVD[%u] Adding type for $__lldb_class: %s", current_id, ast_dumper.GetCString());
                }
                
                TypeFromParser class_type = CopyClassType(class_user_type, current_id);
                
                if (!class_type.IsValid())
                    return;
                
                TypeSourceInfo *type_source_info = m_ast_context->getTrivialTypeSourceInfo(QualType::getFromOpaquePtr(class_type.GetOpaqueQualType()));
                
                if (!type_source_info)
                    return;
                
                TypedefDecl *typedef_decl = TypedefDecl::Create(*m_ast_context,
                                                                m_ast_context->getTranslationUnitDecl(),
                                                                SourceLocation(),
                                                                SourceLocation(),
                                                                context.m_decl_name.getAsIdentifierInfo(),
                                                                type_source_info);
                
                
                if (!typedef_decl)
                    return;
                
                context.AddNamedDecl(typedef_decl);
                
                if (method_decl->isInstance())
                {
                    // self is a pointer to the object
                    
                    QualType class_pointer_type = method_decl->getASTContext().getPointerType(class_qual_type);
                    
                    TypeFromUser self_user_type(class_pointer_type.getAsOpaquePtr(),
                                                &method_decl->getASTContext());
                    
                    m_struct_vars->m_object_pointer_type = self_user_type;
                }
            }
            else
            {
                // This branch will get hit if we are executing code in the context of a function that
                // claims to have an object pointer (through DW_AT_object_pointer?) but is not formally a
                // method of the class.  In that case, just look up the "this" variable in the the current
                // scope and use its type.
                // FIXME: This code is formally correct, but clang doesn't currently emit DW_AT_object_pointer
                // for C++ so it hasn't actually been tested.
                
                VariableList *vars = frame->GetVariableList(false);
                
                lldb::VariableSP this_var = vars->FindVariable(ConstString("this"));
                
                if (this_var &&
                    this_var->IsInScope(frame) &&
                    this_var->LocationIsValidForFrame (frame))
                {
                    Type *this_type = this_var->GetType();
                    
                    if (!this_type)
                        return;
                    
                    QualType this_qual_type = QualType::getFromOpaquePtr(this_type->GetClangFullType());
                    const PointerType *class_pointer_type = this_qual_type->getAs<PointerType>();
                    
                    if (class_pointer_type)
                    {
                        QualType class_type = class_pointer_type->getPointeeType();
                        
                        if (log)
                        {
                            ASTDumper ast_dumper(this_type->GetClangFullType());
                            log->Printf("  FEVD[%u] Adding type for $__lldb_objc_class: %s", current_id, ast_dumper.GetCString());
                        }
                        
                        TypeFromUser class_user_type (class_type.getAsOpaquePtr(),
                                                        this_type->GetClangAST());
                        AddOneType(context, class_user_type, current_id);
                                    
                                    
                        TypeFromUser this_user_type(this_type->GetClangFullType(),
                                                    this_type->GetClangAST());
                        
                        m_struct_vars->m_object_pointer_type = this_user_type;
                        return;
                    }
                }
            }
            
            return;
        }
        
        static ConstString g_lldb_objc_class_name ("$__lldb_objc_class");
        if (name == g_lldb_objc_class_name)
        {
            // Clang is looking for the type of "*self"
            
            if (!frame)
                return;
         
            SymbolContext sym_ctx = frame->GetSymbolContext(lldb::eSymbolContextFunction);
            
            if (!sym_ctx.function)
                return;
            
            // Get the block that defines the function
            Block *function_block = sym_ctx.GetFunctionBlock();
            
            if (!function_block)
                return;
            
            clang::DeclContext *decl_context = function_block->GetClangDeclContext();
            
            if (!decl_context)
                return;
            
            clang::ObjCMethodDecl *method_decl = llvm::dyn_cast<clang::ObjCMethodDecl>(decl_context);
            
            if (method_decl)
            {
                ObjCInterfaceDecl* self_interface = method_decl->getClassInterface();
                
                if (!self_interface)
                    return;
                
                const clang::Type *interface_type = self_interface->getTypeForDecl();
                
                if (!interface_type)
                    return; // This is unlikely, but we have seen crashes where this occurred
                        
                TypeFromUser class_user_type(QualType(interface_type, 0).getAsOpaquePtr(),
                                             &method_decl->getASTContext());
                
                if (log)
                {
                    ASTDumper ast_dumper(interface_type);
                    log->Printf("  FEVD[%u] Adding type for $__lldb_objc_class: %s", current_id, ast_dumper.GetCString());
                }
                    
                AddOneType(context, class_user_type, current_id);
                                
                if (method_decl->isInstanceMethod())
                {
                    // self is a pointer to the object
                    
                    QualType class_pointer_type = method_decl->getASTContext().getObjCObjectPointerType(QualType(interface_type, 0));
                
                    TypeFromUser self_user_type(class_pointer_type.getAsOpaquePtr(),
                                                &method_decl->getASTContext());
                
                    m_struct_vars->m_object_pointer_type = self_user_type;
                }
                else
                {
                    // self is a Class pointer
                    QualType class_type = method_decl->getASTContext().getObjCClassType();
                    
                    TypeFromUser self_user_type(class_type.getAsOpaquePtr(),
                                                &method_decl->getASTContext());
                    
                    m_struct_vars->m_object_pointer_type = self_user_type;
                }

                return;
            }
            else
            {
                // This branch will get hit if we are executing code in the context of a function that
                // claims to have an object pointer (through DW_AT_object_pointer?) but is not formally a
                // method of the class.  In that case, just look up the "self" variable in the the current
                // scope and use its type.
                
                VariableList *vars = frame->GetVariableList(false);
                
                lldb::VariableSP self_var = vars->FindVariable(ConstString("self"));
                
                if (self_var &&
                    self_var->IsInScope(frame) && 
                    self_var->LocationIsValidForFrame (frame))
                {
                    Type *self_type = self_var->GetType();
                    
                    if (!self_type)
                        return;
                    
                    QualType self_qual_type = QualType::getFromOpaquePtr(self_type->GetClangFullType());
                    
                    if (self_qual_type->isObjCClassType())
                    {
                        return;
                    }
                    else if (self_qual_type->isObjCObjectPointerType())
                    {
                        const ObjCObjectPointerType *class_pointer_type = self_qual_type->getAs<ObjCObjectPointerType>();
                    
                        QualType class_type = class_pointer_type->getPointeeType();
                        
                        if (log)
                        {
                            ASTDumper ast_dumper(self_type->GetClangFullType());
                            log->Printf("  FEVD[%u] Adding type for $__lldb_objc_class: %s", current_id, ast_dumper.GetCString());
                        }
                        
                        TypeFromUser class_user_type (class_type.getAsOpaquePtr(),
                                                      self_type->GetClangAST());
                        
                        AddOneType(context, class_user_type, current_id);
                                    
                        TypeFromUser self_user_type(self_type->GetClangFullType(),
                                                    self_type->GetClangAST());
                        
                        m_struct_vars->m_object_pointer_type = self_user_type;
                        return;
                    }
                }
            }

            return;
        }
        
        // any other $__lldb names should be weeded out now
        if (!::strncmp(name_unique_cstr, "$__lldb", sizeof("$__lldb") - 1))
            return;
        
        do
        {
            if (!target)
                break;
            
            ClangASTContext *scratch_clang_ast_context = target->GetScratchClangASTContext();
            
            if (!scratch_clang_ast_context)
                break;
            
            ASTContext *scratch_ast_context = scratch_clang_ast_context->getASTContext();
            
            if (!scratch_ast_context)
                break;
            
            TypeDecl *ptype_type_decl = m_parser_vars->m_persistent_vars->GetPersistentType(name);
            
            if (!ptype_type_decl)
                break;
            
            Decl *parser_ptype_decl = m_ast_importer->CopyDecl(m_ast_context, scratch_ast_context, ptype_type_decl);
            
            if (!parser_ptype_decl)
                break;
            
            TypeDecl *parser_ptype_type_decl = dyn_cast<TypeDecl>(parser_ptype_decl);
            
            if (!parser_ptype_type_decl)
                break;
            
            if (log)
                log->Printf("  CEDM::FEVD[%u] Found persistent type %s", current_id, name.GetCString());
            
            context.AddNamedDecl(parser_ptype_type_decl);
        } while (0);
        
        ClangExpressionVariableSP pvar_sp(m_parser_vars->m_persistent_vars->GetVariable(name));
        
        if (pvar_sp)
        {
            AddOneVariable(context, pvar_sp, current_id);
            return;
        }
        
        const char *reg_name(&name.GetCString()[1]);
        
        if (m_parser_vars->m_exe_ctx.GetRegisterContext())
        {
            const RegisterInfo *reg_info(m_parser_vars->m_exe_ctx.GetRegisterContext()->GetRegisterInfoByName(reg_name));
            
            if (reg_info)
            {
                if (log)
                    log->Printf("  CEDM::FEVD[%u] Found register %s", current_id, reg_info->name);
                
                AddOneRegister(context, reg_info, current_id);
            }
        }
    }
    else
    {
        ValueObjectSP valobj;
        VariableSP var;
        Error err;
        
        if (frame && !namespace_decl)
        {
            valobj = frame->GetValueForVariableExpressionPath(name_unique_cstr, 
                                                              eNoDynamicValues, 
                                                              StackFrame::eExpressionPathOptionCheckPtrVsMember ||
                                                              StackFrame::eExpressionPathOptionsAllowDirectIVarAccess ||
                                                              StackFrame::eExpressionPathOptionsNoFragileObjcIvar ||
                                                              StackFrame::eExpressionPathOptionsNoSyntheticChildren ||
                                                              StackFrame::eExpressionPathOptionsNoSyntheticArrayRange,
                                                              var,
                                                              err);
            
            // If we found a variable in scope, no need to pull up function names
            if (err.Success() && var)
            {
                AddOneVariable(context, var, valobj, current_id);
                context.m_found.variable = true;
                return;
            }
        }
        
        if (target)
        {
            var = FindGlobalVariable (*target,
                                      module_sp,
                                      name,
                                      &namespace_decl,
                                      NULL);
            
            if (var)
            {
                valobj = ValueObjectVariable::Create(target, var);
                AddOneVariable(context, var, valobj, current_id);
                context.m_found.variable = true;
                return;
            }
        }
        
        if (!context.m_found.variable)
        {
            const bool include_inlines = false;
            const bool append = false;
            
            if (namespace_decl && module_sp)
            {
                const bool include_symbols = false;

                module_sp->FindFunctions(name,
                                         &namespace_decl,
                                         eFunctionNameTypeBase, 
                                         include_symbols,
                                         include_inlines,
                                         append,
                                         sc_list);
            }
            else if (target && !namespace_decl)
            {
                const bool include_symbols = true;
                
                // TODO Fix FindFunctions so that it doesn't return
                //   instance methods for eFunctionNameTypeBase.
                
                target->GetImages().FindFunctions(name,
                                                  eFunctionNameTypeFull,
                                                  include_symbols,
                                                  include_inlines,
                                                  append, 
                                                  sc_list);
            }
            
            if (sc_list.GetSize())
            {
                Symbol *generic_symbol = NULL;
                Symbol *non_extern_symbol = NULL;
                
                for (uint32_t index = 0, num_indices = sc_list.GetSize();
                     index < num_indices;
                     ++index)
                {
                    SymbolContext sym_ctx;
                    sc_list.GetContextAtIndex(index, sym_ctx);
                    
                    if (sym_ctx.function)
                    {
                        clang::DeclContext *decl_ctx = sym_ctx.function->GetClangDeclContext();
                        
                        // Filter out class/instance methods.
                        if (dyn_cast<clang::ObjCMethodDecl>(decl_ctx))
                            continue;
                        if (dyn_cast<clang::CXXMethodDecl>(decl_ctx))
                            continue;
                        
                        // TODO only do this if it's a C function; C++ functions may be
                        // overloaded
                        if (!context.m_found.function_with_type_info)
                            AddOneFunction(context, sym_ctx.function, NULL, current_id);
                        context.m_found.function_with_type_info = true;
                        context.m_found.function = true;
                    }
                    else if (sym_ctx.symbol)
                    {
                        if (sym_ctx.symbol->IsExternal())
                            generic_symbol = sym_ctx.symbol;
                        else
                            non_extern_symbol = sym_ctx.symbol;
                    }
                }
                
                if (!context.m_found.function_with_type_info)
                {
                    if (generic_symbol)
                    {
                        AddOneFunction (context, NULL, generic_symbol, current_id);
                        context.m_found.function = true;
                    }
                    else if (non_extern_symbol)
                    {
                        AddOneFunction (context, NULL, non_extern_symbol, current_id);
                        context.m_found.function = true;
                    }
                }
            }
            
            if (!context.m_found.variable && !namespace_decl)
            {
                // We couldn't find a non-symbol variable for this.  Now we'll hunt for a generic 
                // data symbol, and -- if it is found -- treat it as a variable.
                
                const Symbol *data_symbol = FindGlobalDataSymbol(*target, name);
                
                if (data_symbol)
                {
                    AddOneGenericVariable(context, *data_symbol, current_id);
                    context.m_found.variable = true;
                }
            }
        }
    }
}

static clang_type_t
MaybePromoteToBlockPointerType
(
    ASTContext *ast_context,
    clang_type_t candidate_type
)
{
    if (!candidate_type)
        return candidate_type;
    
    QualType candidate_qual_type = QualType::getFromOpaquePtr(candidate_type);
    
    const PointerType *candidate_pointer_type = dyn_cast<PointerType>(candidate_qual_type);
    
    if (!candidate_pointer_type)
        return candidate_type;
    
    QualType pointee_qual_type = candidate_pointer_type->getPointeeType();
    
    const RecordType *pointee_record_type = dyn_cast<RecordType>(pointee_qual_type);
    
    if (!pointee_record_type)
        return candidate_type;
    
    RecordDecl *pointee_record_decl = pointee_record_type->getDecl();
    
    if (!pointee_record_decl->isRecord())
        return candidate_type;
    
    if (!pointee_record_decl->getName().startswith(llvm::StringRef("__block_literal_")))
        return candidate_type;
    
    QualType generic_function_type = ast_context->getFunctionNoProtoType(ast_context->UnknownAnyTy);
    QualType block_pointer_type = ast_context->getBlockPointerType(generic_function_type);
    
    return block_pointer_type.getAsOpaquePtr();
}

Value *
ClangExpressionDeclMap::GetVariableValue
(
    VariableSP &var,
    ASTContext *parser_ast_context,
    TypeFromUser *user_type,
    TypeFromParser *parser_type
)
{
    Log *log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_EXPRESSIONS));
    
    Type *var_type = var->GetType();
    
    if (!var_type) 
    {
        if (log)
            log->PutCString("Skipped a definition because it has no type");
        return NULL;
    }
    
    clang_type_t var_opaque_type = var_type->GetClangFullType();
    
    if (!var_opaque_type)
    {
        if (log)
            log->PutCString("Skipped a definition because it has no Clang type");
        return NULL;
    }
    
    ASTContext *ast = var_type->GetClangASTContext().getASTContext();
    
    if (!ast)
    {
        if (log)
            log->PutCString("There is no AST context for the current execution context");
        return NULL;
    }

    // commented out because of <rdar://problem/11024417>
    //var_opaque_type = MaybePromoteToBlockPointerType (ast, var_opaque_type);
    
    DWARFExpression &var_location_expr = var->LocationExpression();
    
    std::unique_ptr<Value> var_location(new Value);
    
    lldb::addr_t loclist_base_load_addr = LLDB_INVALID_ADDRESS;
    
    Target *target = m_parser_vars->m_exe_ctx.GetTargetPtr();

    if (var_location_expr.IsLocationList())
    {
        SymbolContext var_sc;
        var->CalculateSymbolContext (&var_sc);
        loclist_base_load_addr = var_sc.function->GetAddressRange().GetBaseAddress().GetLoadAddress (target);
    }
    Error err;
    
    if (var->GetLocationIsConstantValueData())
    {
        DataExtractor const_value_extractor;
        
        if (var_location_expr.GetExpressionData(const_value_extractor))
        {
            var_location->operator=(Value(const_value_extractor.GetDataStart(), const_value_extractor.GetByteSize()));
            var_location->SetValueType(Value::eValueTypeHostAddress);
        }
        else
        {
            if (log)
                log->Printf("Error evaluating constant variable: %s", err.AsCString());
            return NULL;
        }
    }
    else if (!var_location_expr.Evaluate(&m_parser_vars->m_exe_ctx, ast, NULL, NULL, NULL, loclist_base_load_addr, NULL, *var_location.get(), &err))
    {
        if (log)
            log->Printf("Error evaluating location: %s", err.AsCString());
        return NULL;
    }
        
    void *type_to_use = NULL;
    
    if (parser_ast_context)
    {
        type_to_use = GuardedCopyType(parser_ast_context, ast, var_opaque_type);
        
        if (!type_to_use)
        {
            if (log)
                log->Printf("Couldn't copy a variable's type into the parser's AST context");
            
            return NULL;
        }
        
        if (parser_type)
            *parser_type = TypeFromParser(type_to_use, parser_ast_context);
    }
    else
        type_to_use = var_opaque_type;
    
    if (var_location.get()->GetContextType() == Value::eContextTypeInvalid)
        var_location.get()->SetContext(Value::eContextTypeClangType, type_to_use);
    
    if (var_location.get()->GetValueType() == Value::eValueTypeFileAddress)
    {
        SymbolContext var_sc;
        var->CalculateSymbolContext(&var_sc);
        
        if (!var_sc.module_sp)
            return NULL;
        
        ObjectFile *object_file = var_sc.module_sp->GetObjectFile();
        
        if (!object_file)
            return NULL;
        
        Address so_addr(var_location->GetScalar().ULongLong(), object_file->GetSectionList());
        
        lldb::addr_t load_addr = so_addr.GetLoadAddress(target);
        
        if (load_addr != LLDB_INVALID_ADDRESS)
        {
            var_location->GetScalar() = load_addr;
            var_location->SetValueType(Value::eValueTypeLoadAddress);
        }
    }
    
    if (user_type)
        *user_type = TypeFromUser(var_opaque_type, ast);
    
    return var_location.release();
}

void
ClangExpressionDeclMap::AddOneVariable (NameSearchContext &context, VariableSP var, ValueObjectSP valobj, unsigned int current_id)
{
    assert (m_parser_vars.get());
    
    Log *log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_EXPRESSIONS));
        
    TypeFromUser ut;
    TypeFromParser pt;
    
    Value *var_location = GetVariableValue (var, 
                                            m_ast_context,
                                            &ut,
                                            &pt);
    
    clang::QualType parser_opaque_type = QualType::getFromOpaquePtr(pt.GetOpaqueQualType());
    
    if (parser_opaque_type.isNull())
        return;
    
    if (const clang::Type *parser_type = parser_opaque_type.getTypePtr())
    {
        if (const TagType *tag_type = dyn_cast<TagType>(parser_type))
            CompleteType(tag_type->getDecl());
    }
    
    if (!var_location)
        return;
    
    NamedDecl *var_decl;
    
    bool is_reference = ClangASTContext::IsReferenceType(pt.GetOpaqueQualType());

    if (is_reference)
        var_decl = context.AddVarDecl(pt.GetOpaqueQualType());
    else
        var_decl = context.AddVarDecl(ClangASTContext::CreateLValueReferenceType(pt.GetASTContext(), pt.GetOpaqueQualType()));
        
    std::string decl_name(context.m_decl_name.getAsString());
    ConstString entity_name(decl_name.c_str());
    ClangExpressionVariableSP entity(m_found_entities.CreateVariable (valobj));
    
    assert (entity.get());
    entity->EnableParserVars(GetParserID());
    ClangExpressionVariable::ParserVars *parser_vars = entity->GetParserVars(GetParserID());
    parser_vars->m_parser_type = pt;
    parser_vars->m_named_decl  = var_decl;
    parser_vars->m_llvm_value  = NULL;
    parser_vars->m_lldb_value  = var_location;
    parser_vars->m_lldb_var    = var;
    
    if (is_reference)
        entity->m_flags |= ClangExpressionVariable::EVTypeIsReference;
    
    if (log)
    {
        ASTDumper orig_dumper(ut.GetOpaqueQualType());
        ASTDumper ast_dumper(var_decl);        
        log->Printf("  CEDM::FEVD[%u] Found variable %s, returned %s (original %s)", current_id, decl_name.c_str(), ast_dumper.GetCString(), orig_dumper.GetCString());
    }
}

void
ClangExpressionDeclMap::AddOneVariable(NameSearchContext &context,
                                       ClangExpressionVariableSP &pvar_sp, 
                                       unsigned int current_id)
{
    Log *log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_EXPRESSIONS));
    
    TypeFromUser user_type (pvar_sp->GetTypeFromUser());
    
    TypeFromParser parser_type (GuardedCopyType(m_ast_context, 
                                                user_type.GetASTContext(), 
                                                user_type.GetOpaqueQualType()),
                                m_ast_context);
    
    if (!parser_type.GetOpaqueQualType())
    {
        if (log)
            log->Printf("  CEDM::FEVD[%u] Couldn't import type for pvar %s", current_id, pvar_sp->GetName().GetCString());
        return;
    }
    
    NamedDecl *var_decl = context.AddVarDecl(ClangASTContext::CreateLValueReferenceType(parser_type.GetASTContext(), parser_type.GetOpaqueQualType()));
    
    pvar_sp->EnableParserVars(GetParserID());
    ClangExpressionVariable::ParserVars *parser_vars = pvar_sp->GetParserVars(GetParserID());
    parser_vars->m_parser_type = parser_type;
    parser_vars->m_named_decl  = var_decl;
    parser_vars->m_llvm_value  = NULL;
    parser_vars->m_lldb_value  = NULL;
    
    if (log)
    {
        ASTDumper ast_dumper(var_decl);
        log->Printf("  CEDM::FEVD[%u] Added pvar %s, returned %s", current_id, pvar_sp->GetName().GetCString(), ast_dumper.GetCString());
    }
}

void
ClangExpressionDeclMap::AddOneGenericVariable(NameSearchContext &context, 
                                              const Symbol &symbol,
                                              unsigned int current_id)
{
    assert(m_parser_vars.get());
    
    Log *log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_EXPRESSIONS));
    
    Target *target = m_parser_vars->m_exe_ctx.GetTargetPtr();

    if (target == NULL)
        return;

    ASTContext *scratch_ast_context = target->GetScratchClangASTContext()->getASTContext();
    
    TypeFromUser user_type (ClangASTContext::CreateLValueReferenceType(scratch_ast_context, ClangASTContext::GetVoidPtrType(scratch_ast_context, false)),
                            scratch_ast_context);
    
    TypeFromParser parser_type (ClangASTContext::CreateLValueReferenceType(m_ast_context, ClangASTContext::GetVoidPtrType(m_ast_context, false)),
                                m_ast_context);
    
    NamedDecl *var_decl = context.AddVarDecl(parser_type.GetOpaqueQualType());
    
    std::string decl_name(context.m_decl_name.getAsString());
    ConstString entity_name(decl_name.c_str());
    ClangExpressionVariableSP entity(m_found_entities.CreateVariable (m_parser_vars->m_exe_ctx.GetBestExecutionContextScope (),
                                                                      entity_name, 
                                                                      user_type,
                                                                      m_parser_vars->m_target_info.byte_order,
                                                                      m_parser_vars->m_target_info.address_byte_size));
    assert (entity.get());
    
    std::unique_ptr<Value> symbol_location(new Value);
    
    const Address &symbol_address = symbol.GetAddress();
    lldb::addr_t symbol_load_addr = symbol_address.GetLoadAddress(target);
    
    symbol_location->SetContext(Value::eContextTypeClangType, user_type.GetOpaqueQualType());
    symbol_location->GetScalar() = symbol_load_addr;
    symbol_location->SetValueType(Value::eValueTypeLoadAddress);
    
    entity->EnableParserVars(GetParserID());
    ClangExpressionVariable::ParserVars *parser_vars = entity->GetParserVars(GetParserID());
    parser_vars->m_parser_type = parser_type;
    parser_vars->m_named_decl  = var_decl;
    parser_vars->m_llvm_value  = NULL;
    parser_vars->m_lldb_value  = symbol_location.release();
    parser_vars->m_lldb_sym    = &symbol;
    
    if (log)
    {
        ASTDumper ast_dumper(var_decl);
        
        log->Printf("  CEDM::FEVD[%u] Found variable %s, returned %s", current_id, decl_name.c_str(), ast_dumper.GetCString());
    }
}

bool 
ClangExpressionDeclMap::ResolveUnknownTypes()
{
    Log *log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_EXPRESSIONS));
    Target *target = m_parser_vars->m_exe_ctx.GetTargetPtr();

    ASTContext *scratch_ast_context = target->GetScratchClangASTContext()->getASTContext();

    for (size_t index = 0, num_entities = m_found_entities.GetSize();
         index < num_entities;
         ++index)
    {
        ClangExpressionVariableSP entity = m_found_entities.GetVariableAtIndex(index);
        
        ClangExpressionVariable::ParserVars *parser_vars = entity->GetParserVars(GetParserID());
        
        if (entity->m_flags & ClangExpressionVariable::EVUnknownType)
        {
            const NamedDecl *named_decl = parser_vars->m_named_decl;
            const VarDecl *var_decl = dyn_cast<VarDecl>(named_decl);
            
            if (!var_decl)
            {
                if (log)
                    log->Printf("Entity of unknown type does not have a VarDecl");
                return false;
            }
            
            if (log)
            {
                ASTDumper ast_dumper(const_cast<VarDecl*>(var_decl));
                log->Printf("Variable of unknown type now has Decl %s", ast_dumper.GetCString());
            }
                
            QualType var_type = var_decl->getType();
            TypeFromParser parser_type(var_type.getAsOpaquePtr(), &var_decl->getASTContext());
            
            lldb::clang_type_t copied_type = m_ast_importer->CopyType(scratch_ast_context, &var_decl->getASTContext(), var_type.getAsOpaquePtr());
            
            if (!copied_type)
            {                
                if (log)
                    log->Printf("ClangExpressionDeclMap::ResolveUnknownType - Couldn't import the type for a variable");
                
                return (bool) lldb::ClangExpressionVariableSP();
            }
            
            TypeFromUser user_type(copied_type, scratch_ast_context);
                        
            parser_vars->m_lldb_value->SetContext(Value::eContextTypeClangType, user_type.GetOpaqueQualType());
            parser_vars->m_parser_type = parser_type;
            
            entity->SetClangAST(user_type.GetASTContext());
            entity->SetClangType(user_type.GetOpaqueQualType());
            
            entity->m_flags &= ~(ClangExpressionVariable::EVUnknownType);
        }
    }
            
    return true;
}

void
ClangExpressionDeclMap::AddOneRegister (NameSearchContext &context,
                                        const RegisterInfo *reg_info, 
                                        unsigned int current_id)
{
    Log *log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_EXPRESSIONS));
    
    void *ast_type = ClangASTContext::GetBuiltinTypeForEncodingAndBitSize(m_ast_context,
                                                                          reg_info->encoding,
                                                                          reg_info->byte_size * 8);
    
    if (!ast_type)
    {
        if (log)
            log->Printf("  Tried to add a type for %s, but couldn't get one", context.m_decl_name.getAsString().c_str());
        return;
    }
    
    TypeFromParser parser_type (ast_type,
                                m_ast_context);
    
    NamedDecl *var_decl = context.AddVarDecl(parser_type.GetOpaqueQualType());
    
    ClangExpressionVariableSP entity(m_found_entities.CreateVariable (m_parser_vars->m_exe_ctx.GetBestExecutionContextScope(),
                                                                      m_parser_vars->m_target_info.byte_order,
                                                                      m_parser_vars->m_target_info.address_byte_size));
    assert (entity.get());
    
    std::string decl_name(context.m_decl_name.getAsString());
    entity->SetName (ConstString (decl_name.c_str()));
    entity->SetRegisterInfo (reg_info);
    entity->EnableParserVars(GetParserID());
    ClangExpressionVariable::ParserVars *parser_vars = entity->GetParserVars(GetParserID());
    parser_vars->m_parser_type = parser_type;
    parser_vars->m_named_decl  = var_decl;
    parser_vars->m_llvm_value  = NULL;
    parser_vars->m_lldb_value  = NULL;
    entity->m_flags |= ClangExpressionVariable::EVBareRegister;
    
    if (log)
    {
        ASTDumper ast_dumper(var_decl);
        log->Printf("  CEDM::FEVD[%d] Added register %s, returned %s", current_id, context.m_decl_name.getAsString().c_str(), ast_dumper.GetCString());
    }
}

void
ClangExpressionDeclMap::AddOneFunction (NameSearchContext &context,
                                        Function* fun,
                                        Symbol* symbol,
                                        unsigned int current_id)
{
    assert (m_parser_vars.get());
    
    Log *log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_EXPRESSIONS));
    
    NamedDecl *fun_decl = NULL;
    std::unique_ptr<Value> fun_location(new Value);
    const Address *fun_address = NULL;
    
    // only valid for Functions, not for Symbols
    void *fun_opaque_type = NULL;
    ASTContext *fun_ast_context = NULL;

    bool is_indirect_function = false;

    if (fun)
    {
        Type *fun_type = fun->GetType();
        
        if (!fun_type) 
        {
            if (log)
                log->PutCString("  Skipped a function because it has no type");
            return;
        }
        
        fun_opaque_type = fun_type->GetClangFullType();
        
        if (!fun_opaque_type)
        {
            if (log)
                log->PutCString("  Skipped a function because it has no Clang type");
            return;
        }
        
        fun_address = &fun->GetAddressRange().GetBaseAddress();
        
        fun_ast_context = fun_type->GetClangASTContext().getASTContext();
        void *copied_type = GuardedCopyType(m_ast_context, fun_ast_context, fun_opaque_type);
        if (copied_type)
        {
            fun_decl = context.AddFunDecl(copied_type);
        }
        else
        {
            // We failed to copy the type we found
            if (log)
            {
                log->Printf ("  Failed to import the function type '%s' {0x%8.8" PRIx64 "} into the expression parser AST contenxt",
                             fun_type->GetName().GetCString(), 
                             fun_type->GetID());
            }
            
            return;
        }
    }
    else if (symbol)
    {
        fun_address = &symbol->GetAddress();
        fun_decl = context.AddGenericFunDecl();
        is_indirect_function = symbol->IsIndirect();
    }
    else
    {
        if (log)
            log->PutCString("  AddOneFunction called with no function and no symbol");
        return;
    }
    
    Target *target = m_parser_vars->m_exe_ctx.GetTargetPtr();

    lldb::addr_t load_addr = fun_address->GetCallableLoadAddress(target, is_indirect_function);
    
    if (load_addr != LLDB_INVALID_ADDRESS)
    {
        fun_location->SetValueType(Value::eValueTypeLoadAddress);
        fun_location->GetScalar() = load_addr;
    }
    else
    {
        // We have to try finding a file address.
        
        lldb::addr_t file_addr = fun_address->GetFileAddress();
        
        fun_location->SetValueType(Value::eValueTypeFileAddress);
        fun_location->GetScalar() = file_addr;
    }
    
    ClangExpressionVariableSP entity(m_found_entities.CreateVariable (m_parser_vars->m_exe_ctx.GetBestExecutionContextScope (),
                                                                      m_parser_vars->m_target_info.byte_order,
                                                                      m_parser_vars->m_target_info.address_byte_size));
    assert (entity.get());
    std::string decl_name(context.m_decl_name.getAsString());
    entity->SetName(ConstString(decl_name.c_str()));
    entity->SetClangType (fun_opaque_type);
    entity->SetClangAST (fun_ast_context);
    
    entity->EnableParserVars(GetParserID());
    ClangExpressionVariable::ParserVars *parser_vars = entity->GetParserVars(GetParserID());
    parser_vars->m_named_decl  = fun_decl;
    parser_vars->m_llvm_value  = NULL;
    parser_vars->m_lldb_value  = fun_location.release();
        
    if (log)
    {
        ASTDumper ast_dumper(fun_decl);
        
        StreamString ss;
        
        fun_address->Dump(&ss, m_parser_vars->m_exe_ctx.GetBestExecutionContextScope(), Address::DumpStyleResolvedDescription);
        
        log->Printf("  CEDM::FEVD[%u] Found %s function %s (description %s), returned %s",
                    current_id,
                    (fun ? "specific" : "generic"), 
                    decl_name.c_str(),
                    ss.GetData(),
                    ast_dumper.GetCString());
    }
}

TypeFromParser
ClangExpressionDeclMap::CopyClassType(TypeFromUser &ut,
                                      unsigned int current_id)
{
    ASTContext *parser_ast_context = m_ast_context;
    ASTContext *user_ast_context = ut.GetASTContext();

    void *copied_type = GuardedCopyType(parser_ast_context, user_ast_context, ut.GetOpaqueQualType());
    
    if (!copied_type)
    {
        Log *log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_EXPRESSIONS));
        
        if (log)
            log->Printf("ClangExpressionDeclMap::CopyClassType - Couldn't import the type");
        
        return TypeFromParser();
    }

    if (ClangASTContext::IsAggregateType(copied_type) && ClangASTContext::GetCompleteType (parser_ast_context, copied_type))
    {
        void *args[1];
        
        args[0] = ClangASTContext::GetVoidPtrType(parser_ast_context, false);
        
        clang_type_t method_type = ClangASTContext::CreateFunctionType (parser_ast_context,
                                                                        ClangASTContext::GetBuiltInType_void(parser_ast_context),
                                                                        args,
                                                                        1,
                                                                        false,
                                                                        ClangASTContext::GetTypeQualifiers(copied_type));
        
        const bool is_virtual = false;
        const bool is_static = false;
        const bool is_inline = false;
        const bool is_explicit = false;
        const bool is_attr_used = true;
        const bool is_artificial = false;
        
        ClangASTContext::AddMethodToCXXRecordType (parser_ast_context,
                                                   copied_type,
                                                   "$__lldb_expr",
                                                   method_type,
                                                   lldb::eAccessPublic,
                                                   is_virtual,
                                                   is_static,
                                                   is_inline,
                                                   is_explicit,
                                                   is_attr_used,
                                                   is_artificial);
    }
    
    return TypeFromParser(copied_type, parser_ast_context);
}

void 
ClangExpressionDeclMap::AddOneType(NameSearchContext &context, 
                                   TypeFromUser &ut,
                                   unsigned int current_id)
{
    ASTContext *parser_ast_context = m_ast_context;
    ASTContext *user_ast_context = ut.GetASTContext();
    
    void *copied_type = GuardedCopyType(parser_ast_context, user_ast_context, ut.GetOpaqueQualType());
    
    if (!copied_type)
    {
        Log *log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_EXPRESSIONS));

        if (log)
            log->Printf("ClangExpressionDeclMap::AddOneType - Couldn't import the type");
        
        return;
    }
    
    context.AddTypeDecl(copied_type);
}
