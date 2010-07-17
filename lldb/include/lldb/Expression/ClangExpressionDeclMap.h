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
    class QualType;
}

namespace llvm {
    class Value;
}

namespace lldb_private {

class Error;
class Function;
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
    
    // Interface for IRForTarget
    bool AddValueToStruct (llvm::Value *value,
                           const clang::NamedDecl *decl,
                           std::string &name,
                           void *type,
                           size_t size,
                           off_t alignment);
    bool DoStructLayout ();
    bool GetStructInfo (uint32_t &num_elements,
                        size_t &size,
                        off_t &alignment);
    bool GetStructElement (const clang::NamedDecl *&decl,
                           llvm::Value *&value,
                           off_t &offset,
                           uint32_t index);
    
    // Interface for DwarfExpression
    Value *GetValueForIndex (uint32_t index);
    
    // Interface for CommandObjectExpression
    lldb::addr_t Materialize(ExecutionContext *exe_ctx,
                             Error &error);
    
    // Interface for ClangASTSource
    void GetDecls (NameSearchContext &context,
                   const char *name);
protected:
private:
    struct Tuple
    {
        const clang::NamedDecl  *m_decl;
        clang::ASTContext       *m_ast_context;
        void                    *m_orig_type;
        Value                   *m_value; /* owned by ClangExpressionDeclMap */
    };
    
    struct StructMember
    {
        const clang::NamedDecl *m_decl;
        llvm::Value            *m_value;
        std::string             m_name;
        void                   *m_type;
        off_t                   m_offset;
        size_t                  m_size;
        off_t                   m_alignment;
    };
    
    typedef std::vector<Tuple> TupleVector;
    typedef TupleVector::iterator TupleIterator;
    
    typedef std::vector<StructMember> StructMemberVector;
    typedef StructMemberVector::iterator StructMemberIterator;
    
    TupleVector         m_tuples;
    StructMemberVector  m_members;
    ExecutionContext   *m_exe_ctx;
    SymbolContext      *m_sym_ctx; /* owned by ClangExpressionDeclMap */
    off_t               m_struct_alignment;
    size_t              m_struct_size;
    bool                m_struct_laid_out;
    lldb::addr_t        m_materialized_location;
        
    Variable *FindVariableInScope(const SymbolContext &sym_ctx,
                                  const char *name,
                                  void *type = NULL,
                                  clang::ASTContext *ast_context = NULL);
    
    Value *GetVariableValue(ExecutionContext &exe_ctx,
                            Variable *var,
                            clang::ASTContext *target_ast_context = NULL,
                            void **opaque_type = NULL,
                            clang::ASTContext **found_ast_context = NULL);
    
    void AddOneVariable(NameSearchContext &context, Variable *var);
    void AddOneFunction(NameSearchContext &context, Function *fun);
    
    bool MaterializeOneVariable(ExecutionContext &exe_ctx,
                                const SymbolContext &sym_ctx,
                                const char *name,
                                void *type,
                                clang::ASTContext *ast_context,
                                lldb::addr_t addr, 
                                Error &err);
};
    
} // namespace lldb_private

#endif  // liblldb_ClangExpressionDeclMap_h_
