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
#include "lldb/lldb-private.h"
#include "lldb/Core/Address.h"
#include "lldb/Core/Error.h"
#include "lldb/Core/Log.h"
#include "lldb/Core/Module.h"
#include "lldb/Expression/ClangASTSource.h"
#include "lldb/Symbol/ClangASTContext.h"
#include "lldb/Symbol/CompileUnit.h"
#include "lldb/Symbol/Function.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Symbol/SymbolContext.h"
#include "lldb/Symbol/Type.h"
#include "lldb/Symbol/TypeList.h"
#include "lldb/Symbol/Variable.h"
#include "lldb/Symbol/VariableList.h"
#include "lldb/Target/ExecutionContext.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/StackFrame.h"
#include "lldb/Target/Target.h"

using namespace lldb_private;
using namespace clang;

ClangExpressionDeclMap::ClangExpressionDeclMap(ExecutionContext *exe_ctx) :
    m_exe_ctx(exe_ctx),
    m_struct_laid_out(false),
    m_materialized_location(0)
{
    if (exe_ctx && exe_ctx->frame)
        m_sym_ctx = new SymbolContext(exe_ctx->frame->GetSymbolContext(lldb::eSymbolContextEverything));
    else
        m_sym_ctx = NULL;
}

ClangExpressionDeclMap::~ClangExpressionDeclMap()
{
    uint32_t num_tuples = m_tuples.size ();
    uint32_t tuple_index;
    
    for (tuple_index = 0; tuple_index < num_tuples; ++tuple_index)
        delete m_tuples[tuple_index].m_value;
    
    if (m_sym_ctx)
        delete m_sym_ctx;
}

bool 
ClangExpressionDeclMap::GetIndexForDecl (uint32_t &index,
                                         const clang::Decl *decl)
{
    uint32_t num_tuples = m_tuples.size ();
    uint32_t tuple_index;
    
    for (tuple_index = 0; tuple_index < num_tuples; ++tuple_index)
    {
        if (m_tuples[tuple_index].m_decl == decl) 
        {
            index = tuple_index;
            return true;
        }
    }
    
    return false;
}

// Interface for IRForTarget

bool 
ClangExpressionDeclMap::AddValueToStruct (llvm::Value *value,
                                          const clang::NamedDecl *decl,
                                          std::string &name,
                                          void *parser_type,
                                          clang::ASTContext *parser_ast_context,
                                          size_t size,
                                          off_t alignment)
{
    m_struct_laid_out = false;
    
    StructMemberIterator iter;
    
    for (iter = m_members.begin();
         iter != m_members.end();
         ++iter)
    {
        if (iter->m_decl == decl)
            return true;
    }

    StructMember member;
    
    member.m_value          = value;
    member.m_decl           = decl;
    member.m_name           = name;
    member.m_parser_type    = TypeFromParser(parser_type, parser_ast_context);
    member.m_offset         = 0;
    member.m_size           = size;
    member.m_alignment      = alignment;
    
    m_members.push_back(member);
    
    return true;
}

bool
ClangExpressionDeclMap::DoStructLayout ()
{
    if (m_struct_laid_out)
        return true;
    
    StructMemberIterator iter;
    
    off_t cursor = 0;
    
    m_struct_alignment = 0;
    m_struct_size = 0;
    
    for (iter = m_members.begin();
         iter != m_members.end();
         ++iter)
    {
        if (iter == m_members.begin())
            m_struct_alignment = iter->m_alignment;
        
        if (cursor % iter->m_alignment)
            cursor += (iter->m_alignment - (cursor % iter->m_alignment));
        
        iter->m_offset = cursor;
        cursor += iter->m_size;
    }
    
    m_struct_size = cursor;
    
    m_struct_laid_out = true;
    return true;
}

bool ClangExpressionDeclMap::GetStructInfo (uint32_t &num_elements,
                                            size_t &size,
                                            off_t &alignment)
{
    if (!m_struct_laid_out)
        return false;
    
    num_elements = m_members.size();
    size = m_struct_size;
    alignment = m_struct_alignment;
    
    return true;
}

bool 
ClangExpressionDeclMap::GetStructElement (const clang::NamedDecl *&decl,
                                          llvm::Value *&value,
                                          off_t &offset,
                                          uint32_t index)
{
    if (!m_struct_laid_out)
        return false;
    
    if (index >= m_members.size())
        return false;
    
    decl = m_members[index].m_decl;
    value = m_members[index].m_value;
    offset = m_members[index].m_offset;
    
    return true;
}

bool
ClangExpressionDeclMap::GetFunctionInfo (const clang::NamedDecl *decl, 
                                         llvm::Value**& value, 
                                         uint64_t &ptr)
{
    TupleIterator iter;
    
    for (iter = m_tuples.begin();
         iter != m_tuples.end();
         ++iter)
    {
        if (decl == iter->m_decl)
        {
            value = &iter->m_llvm_value;
            ptr = iter->m_value->GetScalar().ULongLong();
            return true;
        }
    }
    
    return false;
}

bool
ClangExpressionDeclMap::GetFunctionAddress (const char *name,
                                            uint64_t &ptr)
{
    // Back out in all cases where we're not fully initialized
    if (!m_exe_ctx || !m_exe_ctx->frame || !m_sym_ctx)
        return false;

    ConstString name_cs(name);
    SymbolContextList sym_ctxs;
    
    m_sym_ctx->FindFunctionsByName(name_cs, false, sym_ctxs);
    
    if (!sym_ctxs.GetSize())
        return false;
    
    SymbolContext sym_ctx;
    sym_ctxs.GetContextAtIndex(0, sym_ctx);
    
    const Address *fun_address;
    
    if (sym_ctx.function)
        fun_address = &sym_ctx.function->GetAddressRange().GetBaseAddress();
    else if (sym_ctx.symbol)
        fun_address = &sym_ctx.symbol->GetAddressRangeRef().GetBaseAddress();
    else
        return false;
    
    ptr = fun_address->GetLoadAddress(m_exe_ctx->process);
    
    return true;
}

// Interface for DwarfExpression
lldb_private::Value 
*ClangExpressionDeclMap::GetValueForIndex (uint32_t index)
{
    if (index >= m_tuples.size ())
        return NULL;
    
    return m_tuples[index].m_value;
}

// Interface for CommandObjectExpression

bool 
ClangExpressionDeclMap::Materialize (ExecutionContext *exe_ctx, 
                                     lldb::addr_t &struct_address,
                                     Error &err)
{
    bool result = DoMaterialize(false, exe_ctx, NULL, err);
    
    if (result)
        struct_address = m_materialized_location;
    
    return result;
}

bool 
ClangExpressionDeclMap::Dematerialize (ExecutionContext *exe_ctx,
                                       lldb_private::Value &result_value,
                                       Error &err)
{
    return DoMaterialize(true, exe_ctx, &result_value, err);
}

bool
ClangExpressionDeclMap::DumpMaterializedStruct(ExecutionContext *exe_ctx, 
                                               Stream &s,
                                               Error &err)
{
    if (!m_struct_laid_out)
    {
        err.SetErrorString("Structure hasn't been laid out yet");
        return false;
    }
    
    if (!exe_ctx)
    {
        err.SetErrorString("Received null execution context");
        return false;
    }
    
    
    if (!exe_ctx->process)
    {
        err.SetErrorString("Couldn't find the process");
        return false;
    }
    
    if (!exe_ctx->target)
    {
        err.SetErrorString("Couldn't find the target");
        return false;
    }
    
    lldb::DataBufferSP data(new DataBufferHeap(m_struct_size, 0));    
    
    Error error;
    if (exe_ctx->process->ReadMemory (m_materialized_location, data->GetBytes(), data->GetByteSize(), error) != data->GetByteSize())
    {
        err.SetErrorStringWithFormat ("Couldn't read struct from the target: %s", error.AsCString());
        return false;
    }
    
    DataExtractor extractor(data, exe_ctx->process->GetByteOrder(), exe_ctx->target->GetArchitecture().GetAddressByteSize());
    
    StructMemberIterator iter;
    
    for (iter = m_members.begin();
         iter != m_members.end();
         ++iter)
    {
        s.Printf("[%s]\n", iter->m_name.c_str());
        
        extractor.Dump(&s,                                          // stream
                       iter->m_offset,                              // offset
                       lldb::eFormatBytesWithASCII,                 // format
                       1,                                           // byte size of individual entries
                       iter->m_size,                                // number of entries
                       16,                                          // entries per line
                       m_materialized_location + iter->m_offset,    // address to print
                       0,                                           // bit size (bitfields only; 0 means ignore)
                       0);                                          // bit alignment (bitfields only; 0 means ignore)
        
        s.PutChar('\n');
    }
    
    return true;
}

bool 
ClangExpressionDeclMap::DoMaterialize (bool dematerialize,
                                       ExecutionContext *exe_ctx,
                                       lldb_private::Value *result_value,
                                       Error &err)
{
    Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_EXPRESSIONS);

    if (!m_struct_laid_out)
    {
        err.SetErrorString("Structure hasn't been laid out yet");
        return LLDB_INVALID_ADDRESS;
    }
    
    if (!exe_ctx)
    {
        err.SetErrorString("Received null execution context");
        return LLDB_INVALID_ADDRESS;
    }
    
    if (!exe_ctx->frame)
    {
        err.SetErrorString("Received null execution frame");
        return LLDB_INVALID_ADDRESS;
    }
    
    const SymbolContext &sym_ctx(exe_ctx->frame->GetSymbolContext(lldb::eSymbolContextEverything));
    
    if (!dematerialize)
    {
        if (m_materialized_location)
        {
            exe_ctx->process->DeallocateMemory(m_materialized_location);
            m_materialized_location = 0;
        }
        
        lldb::addr_t mem = exe_ctx->process->AllocateMemory(m_struct_alignment + m_struct_size, 
                                                            lldb::ePermissionsReadable | lldb::ePermissionsWritable,
                                                            err);
        
        if (mem == LLDB_INVALID_ADDRESS)
            return false;
        
        m_allocated_area = mem;
    }
    
    m_materialized_location = m_allocated_area;
    
    if (m_materialized_location % m_struct_alignment)
    {
        m_materialized_location += (m_struct_alignment - (m_materialized_location % m_struct_alignment));
    }
        
    StructMemberIterator iter;
    
    for (iter = m_members.begin();
         iter != m_members.end();
         ++iter)
    {
        uint32_t tuple_index;
        
        if (!GetIndexForDecl(tuple_index, iter->m_decl)) 
        {
            if (iter->m_name.find("___clang_expr_result") == std::string::npos)
            {
                err.SetErrorStringWithFormat("Unexpected variable %s", iter->m_name.c_str());
                return false;
            }
            
            if (log)
                log->Printf("Found special result variable %s", iter->m_name.c_str());
            
            if (dematerialize)
            {
                clang::ASTContext *context(exe_ctx->target->GetScratchClangASTContext()->getASTContext());
                
                if (!context)
                {
                    err.SetErrorString("Couldn't find a scratch AST context to put the result type into"); 
                }
                
                TypeFromUser copied_type(ClangASTContext::CopyType(context, 
                                                                   iter->m_parser_type.GetASTContext(),
                                                                   iter->m_parser_type.GetOpaqueQualType()),
                                         context);
                
                result_value->SetContext(Value::eContextTypeOpaqueClangQualType, copied_type.GetOpaqueQualType());
                
                result_value->SetValueType(Value::eValueTypeLoadAddress);
                result_value->GetScalar() = (uintptr_t)m_materialized_location + iter->m_offset;
            }
            
            continue;
        }
        
        Tuple &tuple(m_tuples[tuple_index]);
        
        if (!DoMaterializeOneVariable(dematerialize, *exe_ctx, sym_ctx, iter->m_name.c_str(), tuple.m_user_type, m_materialized_location + iter->m_offset, err))
            return false;
    }
    
    return true;
}

bool 
ClangExpressionDeclMap::DoMaterializeOneVariable(bool dematerialize,
                                                 ExecutionContext &exe_ctx,
                                                 const SymbolContext &sym_ctx,
                                                 const char *name,
                                                 TypeFromUser type,
                                                 lldb::addr_t addr, 
                                                 Error &err)
{
    Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_EXPRESSIONS);
    
    Variable *var = FindVariableInScope(sym_ctx, name, &type);
    
    if (!var)
    {
        err.SetErrorStringWithFormat("Couldn't find %s with appropriate type", name);
        return false;
    }
    
    if (log)
        log->Printf("%s %s with type %p", (dematerialize ? "Dematerializing" : "Materializing"), name, type.GetOpaqueQualType());
    
    std::auto_ptr<lldb_private::Value> location_value(GetVariableValue(exe_ctx,
                                                                       var,
                                                                       type.GetASTContext()));
    
    if (!location_value.get())
    {
        err.SetErrorStringWithFormat("Couldn't get value for %s", name);
        return false;
    }
    
    if (location_value->GetValueType() == Value::eValueTypeLoadAddress)
    {
        lldb::addr_t value_addr = location_value->GetScalar().ULongLong();
        
        size_t bit_size = ClangASTType::GetClangTypeBitWidth(type.GetASTContext(), type.GetOpaqueQualType());
        size_t byte_size = bit_size % 8 ? ((bit_size + 8) / 8) : (bit_size / 8);
        
        DataBufferHeap data;
        data.SetByteSize(byte_size);
        
        lldb::addr_t src_addr;
        lldb::addr_t dest_addr;
        
        if (dematerialize)
        {
            src_addr = addr;
            dest_addr = value_addr;
        }
        else
        {
            src_addr = value_addr;
            dest_addr = addr;
        }
        
        Error error;
        if (exe_ctx.process->ReadMemory (src_addr, data.GetBytes(), byte_size, error) != byte_size)
        {
            err.SetErrorStringWithFormat ("Couldn't read a composite type from the target: %s", error.AsCString());
            return false;
        }
        
        if (exe_ctx.process->WriteMemory (dest_addr, data.GetBytes(), byte_size, error) != byte_size)
        {
            err.SetErrorStringWithFormat ("Couldn't write a composite type to the target: %s", error.AsCString());
            return false;
        }
        
        if (log)
            log->Printf("Copied from 0x%llx to 0x%llx", (uint64_t)src_addr, (uint64_t)addr);
    }
    else
    {
        StreamString ss;
        
        location_value->Dump(&ss);
        
        err.SetErrorStringWithFormat("%s has a value of unhandled type: %s", name, ss.GetString().c_str());   
    }
    
    return true;
}

Variable*
ClangExpressionDeclMap::FindVariableInScope(const SymbolContext &sym_ctx,
                                            const char *name,
                                            TypeFromUser *type)
{
    Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_EXPRESSIONS);

    Function *function(m_sym_ctx->function);
    Block *block(m_sym_ctx->block);
    
    if (!function || !block)
    {
        if (log)
            log->Printf("function = %p, block = %p", function, block);
        return NULL;
    }
    
    BlockList& blocks(function->GetBlocks(true));
    
    ConstString name_cs(name);
    
    lldb::user_id_t current_block_id;
    
    for (current_block_id = block->GetID();
         current_block_id != Block::InvalidID;
         current_block_id = blocks.GetParent(current_block_id))
    {
        Block *current_block(blocks.GetBlockByID(current_block_id));
        
        lldb::VariableListSP var_list = current_block->GetVariableList(false, true);
        
        if (!var_list)
            continue;
        
        lldb::VariableSP var = var_list->FindVariable(name_cs);
        
        if (!var)
            continue;
        
        // var->GetType()->GetClangAST() is the program's AST context and holds
        // var->GetType()->GetOpaqueClangQualType().
        
        // type is m_type for one of the struct members, which was added by 
        // AddValueToStruct.  That type was extracted from the AST context of
        // the compiler in IRForTarget.  The original for the type was copied
        // out of the program's AST context by AddOneVariable.
        
        // So that we can compare these two without having to copy back
        // something we already had in the original AST context, we maintain 
        // m_orig_type and m_ast_context (which are passed into
        // MaterializeOneVariable by Materialize) for each variable.
        
        if (!type)
            return var.get();
        
        if (type->GetASTContext() == var->GetType()->GetClangAST())
        {
            if (!ClangASTContext::AreTypesSame(type->GetASTContext(), type->GetOpaqueQualType(), var->GetType()->GetOpaqueClangQualType()))
                continue;
        }
        else
        {
            if (log)
                log->PutCString("Skipping a candidate variable because of different AST contexts");
            continue;
        }
        
        return var.get();
    }
    
    {
        CompileUnit *compile_unit = m_sym_ctx->comp_unit;
        
        if (!compile_unit)
        {
            if (log)
                log->Printf("compile_unit = %p", compile_unit);
            return NULL;
        }
        
        lldb::VariableListSP var_list = compile_unit->GetVariableList(true);
        
        if (!var_list)
            return NULL;
        
        lldb::VariableSP var = var_list->FindVariable(name_cs);
        
        if (!var)
            return NULL;

        if (!type)
            return var.get();
        
        if (type->GetASTContext() == var->GetType()->GetClangAST())
        {
            if (!ClangASTContext::AreTypesSame(type->GetASTContext(), type->GetOpaqueQualType(), var->GetType()->GetOpaqueClangQualType()))
                return NULL;
        }
        else
        {
            if (log)
                log->PutCString("Skipping a candidate variable because of different AST contexts");
            return NULL;
        }
        
        return var.get();
    }
    
    return NULL;
}

// Interface for ClangASTSource
void 
ClangExpressionDeclMap::GetDecls(NameSearchContext &context,
                                 const char *name)
{
    Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_EXPRESSIONS);
    
    if (log)
        log->Printf("Hunting for a definition for %s", name);
    
    // Back out in all cases where we're not fully initialized
    if (!m_exe_ctx || !m_exe_ctx->frame || !m_sym_ctx)
        return;
    
    Function *function = m_sym_ctx->function;
    
    if (!function)
    {
        if (log)
            log->Printf("Can't evaluate an expression when not in a function");
        return;
    }
    
    ConstString name_cs(name);
    SymbolContextList sym_ctxs;
    
    m_sym_ctx->FindFunctionsByName(name_cs, false, sym_ctxs);
    
    for (uint32_t index = 0, num_indices = sym_ctxs.GetSize();
         index < num_indices;
         ++index)
    {
        SymbolContext sym_ctx;
        sym_ctxs.GetContextAtIndex(index, sym_ctx);
        
        if (sym_ctx.function)
            AddOneFunction(context, sym_ctx.function, NULL);
        else if(sym_ctx.symbol)
            AddOneFunction(context, NULL, sym_ctx.symbol);
    }
    
    Variable *var = FindVariableInScope(*m_sym_ctx, name);
    
    if (var)
        AddOneVariable(context, var);
    
    /* Commented out pending resolution of a loop when the TagType is imported
    lldb::TypeSP type = m_sym_ctx->FindTypeByName(name_cs);
    
    if (type.get())
        AddOneType(context, type.get());
    */
}
        
Value *
ClangExpressionDeclMap::GetVariableValue(ExecutionContext &exe_ctx,
                                         Variable *var,
                                         clang::ASTContext *parser_ast_context,
                                         TypeFromUser *user_type,
                                         TypeFromParser *parser_type)
{
    Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_EXPRESSIONS);
    
    Type *var_type = var->GetType();
    
    if (!var_type) 
    {
        if (log)
            log->PutCString("Skipped a definition because it has no type");
        return NULL;
    }
    
    void *var_opaque_type = var_type->GetOpaqueClangQualType();
    
    if (!var_opaque_type)
    {
        if (log)
            log->PutCString("Skipped a definition because it has no Clang type");
        return NULL;
    }
    
    TypeList *type_list = var_type->GetTypeList();
    
    if (!type_list)
    {
        if (log)
            log->PutCString("Skipped a definition because the type has no associated type list");
        return NULL;
    }
    
    clang::ASTContext *exe_ast_ctx = type_list->GetClangASTContext().getASTContext();
    
    if (!exe_ast_ctx)
    {
        if (log)
            log->PutCString("There is no AST context for the current execution context");
        return NULL;
    }
    
    DWARFExpression &var_location_expr = var->LocationExpression();
    
    std::auto_ptr<Value> var_location(new Value);
    
    Error err;
    
    if (!var_location_expr.Evaluate(&exe_ctx, exe_ast_ctx, NULL, *var_location.get(), &err))
    {
        if (log)
            log->Printf("Error evaluating location: %s", err.AsCString());
        return NULL;
    }
    
    clang::ASTContext *var_ast_context = type_list->GetClangASTContext().getASTContext();
    
    void *type_to_use;
    
    if (parser_ast_context)
    {
        type_to_use = ClangASTContext::CopyType(parser_ast_context, var_ast_context, var_opaque_type);
        
        if (parser_type)
            *parser_type = TypeFromParser(type_to_use, parser_ast_context);
    }
    else
        type_to_use = var_opaque_type;
    
    if (var_location.get()->GetContextType() == Value::eContextTypeInvalid)
        var_location.get()->SetContext(Value::eContextTypeOpaqueClangQualType, type_to_use);
    
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
        
        lldb::addr_t load_addr = so_addr.GetLoadAddress(m_exe_ctx->process);
        
        var_location->GetScalar() = load_addr;
        var_location->SetValueType(Value::eValueTypeLoadAddress);
    }
    
    if (user_type)
        *user_type = TypeFromUser(var_opaque_type, var_ast_context);
    
    return var_location.release();
}

void
ClangExpressionDeclMap::AddOneVariable(NameSearchContext &context,
                                       Variable* var)
{
    Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_EXPRESSIONS);
    
    TypeFromUser ut;
    TypeFromParser pt;
    
    Value *var_location = GetVariableValue(*m_exe_ctx, 
                                           var, 
                                           context.GetASTContext(),
                                           &ut,
                                           &pt);
    
    NamedDecl *var_decl = context.AddVarDecl(pt.GetOpaqueQualType());
    
    Tuple tuple;
    
    tuple.m_decl        = var_decl;
    tuple.m_value       = var_location;
    tuple.m_user_type   = ut;
    tuple.m_parser_type = pt;
    tuple.m_llvm_value  = NULL;
    
    m_tuples.push_back(tuple);
    
    if (log)
        log->Printf("Found variable %s, returned (NamedDecl)%p", context.Name.getAsString().c_str(), var_decl);    
}

void
ClangExpressionDeclMap::AddOneFunction(NameSearchContext &context,
                                       Function* fun,
                                       Symbol* symbol)
{
    Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_EXPRESSIONS);
    
    NamedDecl *fun_decl;
    std::auto_ptr<Value> fun_location(new Value);
    const Address *fun_address;
    
    // only valid for Functions, not for Symbols
    void *fun_opaque_type = NULL;
    clang::ASTContext *fun_ast_context = NULL;
    
    if (fun)
    {
        Type *fun_type = fun->GetType();
        
        if (!fun_type) 
        {
            if (log)
                log->PutCString("Skipped a function because it has no type");
            return;
        }
        
        fun_opaque_type = fun_type->GetOpaqueClangQualType();
        
        if (!fun_opaque_type)
        {
            if (log)
                log->PutCString("Skipped a function because it has no Clang type");
            return;
        }
        
        fun_address = &fun->GetAddressRange().GetBaseAddress();
        
        TypeList *type_list = fun_type->GetTypeList();
        fun_ast_context = type_list->GetClangASTContext().getASTContext();
        void *copied_type = ClangASTContext::CopyType(context.GetASTContext(), fun_ast_context, fun_opaque_type);
        
        fun_decl = context.AddFunDecl(copied_type);
    }
    else if (symbol)
    {
        fun_address = &symbol->GetAddressRangeRef().GetBaseAddress();
        
        fun_decl = context.AddGenericFunDecl();
    }
    else
    {
        if (log)
            log->PutCString("AddOneFunction called with no function and no symbol");
        return;
    }
    
    lldb::addr_t load_addr = fun_address->GetLoadAddress(m_exe_ctx->process);
    fun_location->SetValueType(Value::eValueTypeLoadAddress);
    fun_location->GetScalar() = load_addr;
    
    Tuple tuple;
    
    tuple.m_decl        = fun_decl;
    tuple.m_value       = fun_location.release();
    tuple.m_user_type   = TypeFromUser(fun_opaque_type, fun_ast_context);
    tuple.m_llvm_value  = NULL;
    
    m_tuples.push_back(tuple);
    
    if (log)
        log->Printf("Found function %s, returned (NamedDecl)%p", context.Name.getAsString().c_str(), fun_decl);    
}

void 
ClangExpressionDeclMap::AddOneType(NameSearchContext &context, 
                                   Type *type)
{
    TypeFromUser ut(type->GetOpaqueClangQualType(),
                    type->GetClangAST());
    
    void *copied_type = ClangASTContext::CopyType(context.GetASTContext(), ut.GetASTContext(), ut.GetOpaqueQualType());
    
    context.AddTypeDecl(copied_type);
}
