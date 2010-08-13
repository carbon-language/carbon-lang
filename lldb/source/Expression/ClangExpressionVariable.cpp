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

using namespace lldb_private;
using namespace clang;

ClangExpressionVariableList::ClangExpressionVariableList() :
    m_variables()
{
}

ClangExpressionVariableList::~ClangExpressionVariableList()
{
    uint32_t num_variables = m_variables.size();
    uint32_t var_index;
        
    for (var_index = 0; var_index < num_variables; ++var_index)
        delete m_variables[var_index].m_value;
}

Value *
ValueForDecl(const VarDecl *var_decl)
{
    Value *ret = new Value;
        
    ret->SetContext(Value::eContextTypeOpaqueClangQualType, 
                    var_decl->getType().getAsOpaquePtr());
    
    uint64_t bit_width = var_decl->getASTContext().getTypeSize(var_decl->getType());
    
    uint32_t byte_size = (bit_width + 7 ) / 8;
    
    ret->ResizeData(byte_size);
    
    return ret;
}

Value *
ClangExpressionVariableList::GetVariableForVarDecl (const VarDecl *var_decl, uint32_t& idx, bool can_create)
{
    uint32_t num_variables = m_variables.size();
    uint32_t var_index;
    
    for (var_index = 0; var_index < num_variables; ++var_index)
    {
        if (m_variables[var_index].m_var_decl == var_decl)
        {
            idx = var_index;
            return m_variables[var_index].m_value;
        }
    }

    if (!can_create)
        return NULL;
    
    idx = m_variables.size();
    
    ClangExpressionVariable val;
    val.m_var_decl = var_decl;
    val.m_value = ValueForDecl(var_decl);
    m_variables.push_back(val);
    
    return m_variables.back().m_value;
}

Value *
ClangExpressionVariableList::GetVariableAtIndex (uint32_t idx)
{
    if (idx < m_variables.size())
        return m_variables[idx].m_value;
    
    return NULL;
}
