//===-- ClangExpressionVariable.cpp -----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Expression/ClangExpressionVariable.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "clang/AST/ASTContext.h"
#include "lldb/Core/ConstString.h"
#include "lldb/Core/DataExtractor.h"
#include "lldb/Core/Stream.h"
#include "lldb/Core/Value.h"
#include "lldb/Core/ValueObjectConstResult.h"
#include "lldb/Target/ExecutionContext.h"
#include "lldb/Target/Process.h"

using namespace lldb_private;
using namespace clang;

ClangExpressionVariable::ClangExpressionVariable() :
    m_name(),
    m_user_type (TypeFromUser(NULL, NULL)),
    m_store (NULL),
    m_index (0),
    m_parser_vars(),
    m_jit_vars (),
    m_data_sp ()
{
}

void 
ClangExpressionVariable::DisableDataVars()
{
    m_data_sp.reset();
}


ClangExpressionVariable::ClangExpressionVariable(const ClangExpressionVariable &rhs) :
    m_name(rhs.m_name),
    m_user_type(rhs.m_user_type),
    m_store(rhs.m_store),
    m_index(rhs.m_index)
{
    if (rhs.m_parser_vars.get())
    {
        // TODO: Sean, can m_parser_vars be a shared pointer??? We are copy
        // constructing it here. That is ok if we need to, but do we really
        // need to?
        m_parser_vars.reset(new struct ParserVars);
        *m_parser_vars.get() = *rhs.m_parser_vars.get();
    }
    
    if (rhs.m_jit_vars.get())
    {
        // TODO: Sean, can m_jit_vars be a shared pointer??? We are copy
        // constructing it here. That is ok if we need to, but do we really
        // need to?
        m_jit_vars.reset(new struct JITVars);
        *m_jit_vars.get() = *rhs.m_jit_vars.get();
    }
    
    if (rhs.m_data_sp)
    {
        // TODO: Sean, does m_data_sp need to be copy constructed? Or can it
        // shared the data?
        
        m_data_sp.reset(new DataBufferHeap (rhs.m_data_sp->GetBytes(),
                                            rhs.m_data_sp->GetByteSize()));
    }
}

bool
ClangExpressionVariable::PointValueAtData(Value &value, ExecutionContext *exe_ctx)
{
    if (m_data_sp.get() == NULL)
        return false;
    
    value.SetContext(Value::eContextTypeClangType, m_user_type.GetOpaqueQualType());
    value.SetValueType(Value::eValueTypeHostAddress);
    value.GetScalar() = (uintptr_t)m_data_sp->GetBytes();
    clang::ASTContext *ast_context = m_user_type.GetASTContext();

    if (exe_ctx)
        value.ResolveValue (exe_ctx, ast_context);
    
    return true;
}

void 
ClangExpressionVariable::EnableDataVars()
{
    if (!m_data_sp.get())
        m_data_sp.reset(new DataBufferHeap);
}

lldb::ValueObjectSP
ClangExpressionVariable::GetExpressionResult (ExecutionContext *exe_ctx)
{
    lldb::ValueObjectSP result_sp;
    if (m_data_sp)
    {
        Target * target = NULL;
        Process *process = NULL;
        if (exe_ctx)
        {
            target = exe_ctx->target;
            process = exe_ctx->process;
        }
        
        Value value;
        if (PointValueAtData(value, exe_ctx))
        {
            lldb::ByteOrder byte_order = lldb::eByteOrderHost;
            uint32_t addr_byte_size = 4;
            if (process)
            {
                byte_order = process->GetByteOrder();
                addr_byte_size = process->GetAddressByteSize();
            }
            result_sp.reset (new ValueObjectConstResult (m_user_type.GetASTContext(),
                                                         m_user_type.GetOpaqueQualType(),
                                                         m_name,
                                                         m_data_sp,// TODO: sean can you get this to be valid?
                                                         byte_order,
                                                         addr_byte_size));
        }
    }
    return result_sp;
}

