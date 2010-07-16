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
#include "lldb/Target/Process.h"
#include "lldb/Target/StackFrame.h"
#include "lldb/Target/ExecutionContext.h"

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
                                          void *type,
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
    
    member.m_value      = value;
    member.m_decl       = decl;
    member.m_name       = name;
    member.m_type       = type;
    member.m_offset     = 0;
    member.m_size       = size;
    member.m_alignment  = alignment;
    
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

// Interface for DwarfExpression
Value 
*ClangExpressionDeclMap::GetValueForIndex (uint32_t index)
{
    if (index >= m_tuples.size ())
        return NULL;
    
    return m_tuples[index].m_value;
}

// Interface for CommandObjectExpression
lldb::addr_t 
ClangExpressionDeclMap::Materialize (ExecutionContext *exe_ctx, Error &err)
{
    if (!m_struct_laid_out)
    {
        err.SetErrorString("Structure hasn't been laid out yet");
        return LLDB_INVALID_ADDRESS;
    }
    
    if (m_materialized_location)
    {
        exe_ctx->process->DeallocateMemory(m_materialized_location);
        m_materialized_location = 0;
    }
    
    if (!exe_ctx)
    {
        err.SetErrorString("Received null execution context");
        return LLDB_INVALID_ADDRESS;
    }
    
    const SymbolContext &sym_ctx(exe_ctx->frame->GetSymbolContext(lldb::eSymbolContextEverything));
    
    StructMemberIterator iter;
        
    lldb::addr_t mem = exe_ctx->process->AllocateMemory(m_struct_alignment + m_struct_size, 
                                                        lldb::ePermissionsReadable | lldb::ePermissionsWritable,
                                                        err);
    
    if (mem == LLDB_INVALID_ADDRESS)
        return LLDB_INVALID_ADDRESS;
    
    m_materialized_location = mem;
    
    lldb::addr_t aligned_mem = mem;
    
    if (aligned_mem % m_struct_alignment)
    {
        aligned_mem += (m_struct_alignment - (aligned_mem % m_struct_alignment));
    }
    
    for (iter = m_members.begin();
         iter != m_members.end();
         ++iter)
    {
        uint32_t tuple_index;
        
        if (!GetIndexForDecl(tuple_index, iter->m_decl))
            continue;
        
        Tuple &tuple(m_tuples[tuple_index]);
        
        MaterializeOneVariable(*exe_ctx, sym_ctx, iter->m_name.c_str(), tuple.m_orig_type, tuple.m_ast_context, aligned_mem + iter->m_offset);
    }
    
    return aligned_mem;
}

bool 
ClangExpressionDeclMap::MaterializeOneVariable(ExecutionContext &exe_ctx,
                                               const SymbolContext &sym_ctx,
                                               const char *name,
                                               void *type,
                                               clang::ASTContext *ast_context,
                                               lldb::addr_t addr)
{
    Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_EXPRESSIONS);

    Function *function(m_sym_ctx->function);
    Block *block(m_sym_ctx->block);
    
    if (!function || !block)
    {
        if (log)
            log->Printf("function = %p, block = %p", function, block);
        return false;
    }
    
    BlockList& blocks(function->GetBlocks(true));
    
    lldb::user_id_t current_block_id;
    
    for (current_block_id = block->GetID();
         current_block_id != Block::InvalidID;
         current_block_id = blocks.GetParent(current_block_id))
    {
        Block *current_block(blocks.GetBlockByID(current_block_id));
        
        lldb::VariableListSP var_list = current_block->GetVariableList(false, true);
        
        if (!var_list)
            continue;
        
        lldb::VariableSP var = var_list->FindVariable(ConstString(name));
        
        if (!var)
            continue;
        
        // var->GetType()->GetClangAST() is the program's AST context and holds
        // var->GetType()->GetOpaqueClangQualType().
        
        // type is m_type for one of the struct members, which was added by 
        // AddValueToStruct.  That type was extracted from the AST context of
        // the compiler in IRForTarget.  The original for the type was copied
        // out of the program's AST context by AddOneVariable.
        
        // The key here is: we know when we copied a type, and for what Decl we
        // did it.  So we need for each struct Tuple to keep the type that we
        // found, and which AST context we found it in. Then we can look up
        // m_decl in m_tuples.
        
        if (ast_context == var->GetType()->GetClangAST())
        {
            if (!ClangASTContext::AreTypesSame(ast_context, type, var->GetType()->GetOpaqueClangQualType()))
                continue;
        }
        else
        {
            if (log)
                log->PutCString("Skipping a candidate variable because of different AST contexts");
            continue;
        }
        
        
    }
    
    return true;
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
    Block *block = m_sym_ctx->block;
    
    if (!function || !block)
    {
        if (log)
            log->Printf("function = %p, block = %p", function, block);
        return;
    }
    
    BlockList& blocks = function->GetBlocks(true);
    
    lldb::user_id_t current_block_id = block->GetID();
    
    ConstString name_cs(name);
    
    Function *fn = m_sym_ctx->FindFunctionByName(name_cs.GetCString());
    
    if (fn)
    {
        AddOneFunction(context, fn);
    }
    
    for (current_block_id = block->GetID();
         current_block_id != Block::InvalidID;
         current_block_id = blocks.GetParent(current_block_id))
    {
        Block *current_block = blocks.GetBlockByID(current_block_id);
        
        lldb::VariableListSP var_list = current_block->GetVariableList(false, true);
        
        if (!var_list)
            continue;
        
        lldb::VariableSP var = var_list->FindVariable(name_cs);
        
        if (!var)
            continue;
        
        AddOneVariable(context, var.get());
        return;
    }
    
    {
        CompileUnit *compile_unit = m_sym_ctx->comp_unit;
        
        if (!compile_unit)
        {
            if (log)
                log->Printf("compile_unit = %p", compile_unit);
            return;
        }
        
        lldb::VariableListSP var_list = compile_unit->GetVariableList(true);
        
        if (!var_list)
            return;
        
        lldb::VariableSP var = var_list->FindVariable(name_cs);
        
        if (!var)
            return;
        
        AddOneVariable(context, var.get());
        return;
    }
}

void
ClangExpressionDeclMap::AddOneVariable(NameSearchContext &context,
                                       Variable* var)
{
    Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_EXPRESSIONS);
    
    Type *var_type = var->GetType();
    
    if (!var_type) 
    {
        if (log)
            log->PutCString("Skipped a definition because it has no type");
        return;
    }
    
    void *var_opaque_type = var_type->GetOpaqueClangQualType();
    
    if (!var_opaque_type)
    {
        if (log)
            log->PutCString("Skipped a definition because it has no Clang type");
        return;
    }
    
    DWARFExpression &var_location_expr = var->LocationExpression();
    
    TypeList *type_list = var_type->GetTypeList();
    
    if (!type_list)
    {
        if (log)
            log->PutCString("Skipped a definition because the type has no associated type list");
        return;
    }
    
    clang::ASTContext *exe_ast_ctx = type_list->GetClangASTContext().getASTContext();
    
    if (!exe_ast_ctx)
    {
        if (log)
            log->PutCString("There is no AST context for the current execution context");
        return;
    }
    
    std::auto_ptr<Value> var_location(new Value);
    
    Error err;
    
    if (!var_location_expr.Evaluate(m_exe_ctx, exe_ast_ctx, NULL, *var_location.get(), &err))
    {
        if (log)
            log->Printf("Error evaluating location: %s", err.AsCString());
        return;
    }
    
    clang::ASTContext *var_ast_context = type_list->GetClangASTContext().getASTContext();
    
    void *copied_type = ClangASTContext::CopyType(context.GetASTContext(), var_ast_context, var_opaque_type);
    
    if (var_location.get()->GetContextType() == Value::eContextTypeInvalid)
        var_location.get()->SetContext(Value::eContextTypeOpaqueClangQualType, copied_type);
    
    if (var_location.get()->GetValueType() == Value::eValueTypeFileAddress)
    {
        SymbolContext var_sc;
        var->CalculateSymbolContext(&var_sc);
    
        if (!var_sc.module_sp)
            return;
        
        ObjectFile *object_file = var_sc.module_sp->GetObjectFile();
        
        if (!object_file)
            return;
    
        Address so_addr(var_location->GetScalar().ULongLong(), object_file->GetSectionList());
        
        lldb::addr_t load_addr = so_addr.GetLoadAddress(m_exe_ctx->process);
        
        var_location->GetScalar() = load_addr;
        var_location->SetValueType(Value::eValueTypeLoadAddress);
    }
    
    NamedDecl *var_decl = context.AddVarDecl(copied_type);
    
    Tuple tuple;
    
    tuple.m_decl        = var_decl;
    tuple.m_value       = var_location.release();
    tuple.m_orig_type   = var_opaque_type;
    tuple.m_ast_context = var_ast_context;
    
    m_tuples.push_back(tuple);
    
    if (log)
        log->PutCString("Found variable");    
}

void
ClangExpressionDeclMap::AddOneFunction(NameSearchContext &context,
                                       Function* fun)
{
    Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_EXPRESSIONS);

    Type *fun_type = fun->GetType();
    
    if (!fun_type) 
    {
        if (log)
            log->PutCString("Skipped a function because it has no type");
        return;
    }
    
    void *fun_opaque_type = fun_type->GetOpaqueClangQualType();
    
    if (!fun_opaque_type)
    {
        if (log)
            log->PutCString("Skipped a function because it has no Clang type");
        return;
    }
    
    std::auto_ptr<Value> fun_location(new Value);
    
    const Address &fun_address = fun->GetAddressRange().GetBaseAddress();
    lldb::addr_t load_addr = fun_address.GetLoadAddress(m_exe_ctx->process);
    fun_location->SetValueType(Value::eValueTypeLoadAddress);
    fun_location->GetScalar() = load_addr;
    
    TypeList *type_list = fun_type->GetTypeList();
    clang::ASTContext *fun_ast_context = type_list->GetClangASTContext().getASTContext();
    void *copied_type = ClangASTContext::CopyType(context.GetASTContext(), fun_ast_context, fun_opaque_type);
    
    NamedDecl *fun_decl = context.AddFunDecl(copied_type);
    
    Tuple tuple;
    
    tuple.m_decl        = fun_decl;
    tuple.m_value       = fun_location.release();
    tuple.m_orig_type   = fun_opaque_type;
    tuple.m_ast_context = fun_ast_context;
    
    m_tuples.push_back(tuple);
    
    if (log)
        log->PutCString("Found function");    
}
