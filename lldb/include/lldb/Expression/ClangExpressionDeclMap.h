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
#include "lldb/Symbol/TaggedASTType.h"

namespace llvm {
    class Type;
    class Value;
}

namespace lldb_private {

class ClangPersistentVariables;
class Error;
class Function;
class NameSearchContext;
class Variable;
    
class ClangExpressionDeclMap
{
public:
    ClangExpressionDeclMap(ExecutionContext *exe_ctx,
                           ClangPersistentVariables &persistent_vars);
    ~ClangExpressionDeclMap();
    
    // Interface for ClangStmtVisitor
    bool GetIndexForDecl (uint32_t &index,
                          const clang::Decl *decl);
    
    // Interface for IRForTarget
    bool AddValueToStruct (llvm::Value *value,
                           const clang::NamedDecl *decl,
                           std::string &name,
                           void *type,
                           clang::ASTContext *ast_context,
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
    
    bool GetFunctionInfo (const clang::NamedDecl *decl, 
                          llvm::Value**& value, 
                          uint64_t &ptr);
    
    bool GetFunctionAddress (const char *name,
                             uint64_t &ptr);
    
    // Interface for DwarfExpression
    Value *GetValueForIndex (uint32_t index);
    
    // Interface for CommandObjectExpression
    bool Materialize(ExecutionContext *exe_ctx,
                     lldb::addr_t &struct_address,
                     Error &error);
    
    bool DumpMaterializedStruct(ExecutionContext *exe_ctx,
                                Stream &s,
                                Error &error);
    
    bool Dematerialize(ExecutionContext *exe_ctx,
                       lldb_private::Value &result_value,
                       Error &error);
    
    // Interface for ClangASTSource
    void GetDecls (NameSearchContext &context,
                   const char *name);

    typedef TaggedASTType<0> TypeFromParser;
    typedef TaggedASTType<1> TypeFromUser;
private:    
    struct Tuple
    {
        const clang::NamedDecl  *m_decl;
        TypeFromParser          m_parser_type;
        TypeFromUser            m_user_type;
        lldb_private::Value     *m_value; /* owned by ClangExpressionDeclMap */
        llvm::Value             *m_llvm_value;
    };
    
    struct StructMember
    {
        const clang::NamedDecl *m_decl;
        llvm::Value            *m_value;
        std::string             m_name;
        TypeFromParser          m_parser_type;
        off_t                   m_offset;
        size_t                  m_size;
        off_t                   m_alignment;
    };
    
    typedef std::vector<Tuple> TupleVector;
    typedef TupleVector::iterator TupleIterator;
    
    typedef std::vector<StructMember> StructMemberVector;
    typedef StructMemberVector::iterator StructMemberIterator;
    
    TupleVector                 m_tuples;
    StructMemberVector          m_members;
    ExecutionContext           *m_exe_ctx;
    SymbolContext              *m_sym_ctx; /* owned by ClangExpressionDeclMap */
    ClangPersistentVariables   &m_persistent_vars;
    off_t                       m_struct_alignment;
    size_t                      m_struct_size;
    bool                        m_struct_laid_out;
    lldb::addr_t                m_allocated_area;
    lldb::addr_t                m_materialized_location;
        
    Variable *FindVariableInScope(const SymbolContext &sym_ctx,
                                  const char *name,
                                  TypeFromUser *type = NULL);
    
    Value *GetVariableValue(ExecutionContext &exe_ctx,
                            Variable *var,
                            clang::ASTContext *parser_ast_context,
                            TypeFromUser *found_type = NULL,
                            TypeFromParser *parser_type = NULL);
    
    void AddOneVariable(NameSearchContext &context, Variable *var);
    void AddOneFunction(NameSearchContext &context, Function *fun, Symbol *sym);
    void AddOneType(NameSearchContext &context, Type *type);
    
    bool DoMaterialize (bool dematerialize,
                        ExecutionContext *exe_ctx,
                        lldb_private::Value *result_value, /* must be non-NULL if D is set */
                        Error &err);

    bool DoMaterializeOneVariable(bool dematerialize,
                                  ExecutionContext &exe_ctx,
                                  const SymbolContext &sym_ctx,
                                  const char *name,
                                  TypeFromUser type,
                                  lldb::addr_t addr, 
                                  Error &err);
};
    
class ClangPersistentVariables
{
    
};
    
} // namespace lldb_private

#endif  // liblldb_ClangExpressionDeclMap_h_
