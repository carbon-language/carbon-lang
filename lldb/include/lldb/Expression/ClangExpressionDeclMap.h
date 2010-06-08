//===-- ClangExpressionDeclMap.h --------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_ClangExpressionDeclMap_h_
#define liblldb_ClangExpressionDeclMap_h_

// C Includes
#include <signal.h>
#include <stdint.h>

// C++ Includes
#include <vector>

// Other libraries and framework includes
// Project includes
#include "lldb/Core/ClangForward.h"
#include "lldb/Core/Value.h"

namespace clang {
    class DeclarationName;
    class DeclContext;
}

namespace lldb_private {

class NameSearchContext;
class Variable;
    
class ClangExpressionDeclMap
{
public:
    ClangExpressionDeclMap(ExecutionContext *exe_ctx);
    ~ClangExpressionDeclMap();
    
    // Interface for ClangStmtVisitor
    bool GetIndexForDecl (uint32_t &index,
                          const clang::Decl *decl);
    
    // Interface for DwarfExpression
    Value *GetValueForIndex (uint32_t index);
    
    // Interface for ClangASTSource
    void GetDecls (NameSearchContext &context,
                   const char *name);
protected:
private:
    struct Tuple
    {
        const clang::NamedDecl  *m_decl;
        Value                   *m_value; /* owned by ClangExpressionDeclMap */
    };
    
    typedef std::vector<Tuple> TupleVector;
    typedef TupleVector::iterator TupleIterator;
    
    TupleVector         m_tuples;
    ExecutionContext   *m_exe_ctx;
    SymbolContext      *m_sym_ctx; /* owned by ClangExpressionDeclMap */
    
    void AddOneVariable(NameSearchContext &context, Variable* var);
};
    
} // namespace lldb_private

#endif  // liblldb_ClangExpressionDeclMap_h_
