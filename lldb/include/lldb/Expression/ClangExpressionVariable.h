//===-- ClangExpressionVariable.h -------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_ClangExpressionVariable_h_
#define liblldb_ClangExpressionVariable_h_

// C Includes
#include <signal.h>
#include <stdint.h>

// C++ Includes
#include <vector>

// Other libraries and framework includes
// Project includes
#include "lldb/Core/ClangForward.h"
#include "lldb/Core/Value.h"

namespace lldb_private {

class ClangExpressionVariableList
{
public:
    ClangExpressionVariableList();
    ~ClangExpressionVariableList();

    Value *
    GetVariableForVarDecl (clang::ASTContext &ast_context, 
                           const clang::VarDecl *var_decl, 
                           uint32_t& idx, 
                           bool can_create);

    Value *
    GetVariableAtIndex (uint32_t idx);
    
    uint32_t
    AppendValue (Value *value); // takes ownership

private:
    struct ClangExpressionVariable
    {
        const clang::VarDecl    *m_var_decl;
        Value                   *m_value;
    };
    
    typedef std::vector<ClangExpressionVariable> Variables;
    Variables m_variables;
};

} // namespace lldb_private

#endif  // liblldb_ClangExpressionVariable_h_
