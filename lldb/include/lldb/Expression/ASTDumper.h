//===-- ASTDumper.h ---------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/DeclVisitor.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/AST/TypeVisitor.h"

#include "lldb/Core/Stream.h"
#include "llvm/ADT/DenseSet.h"

namespace lldb_private
{

//----------------------------------------------------------------------
/// @class ASTDumper ASTDumper.h "lldb/Expression/ASTDumper.h"
/// @brief Encapsulates a recursive dumper for Clang AST nodes.
///
/// ASTDumper contains a variety of methods for printing fields of Clang
/// AST structures, for debugging purposes.  It prints the AST objects
/// hierarchically:
///
/// ---
/// class : InheritedClass
/// someAccessor() : result
/// accessorReturningObject() :
///  class : ChildClass [object returned by accessorReturningObject]
///  ...
/// class : BaseClass [same object as InheritedClass]
/// baseAccessor() : result
///
/// The output format is YAML.
//----------------------------------------------------------------------
class ASTDumper : 
    public clang::DeclVisitor <ASTDumper, void>, 
    public clang::StmtVisitor <ASTDumper, void>,
    public clang::TypeVisitor <ASTDumper, void>
{
private:
    ASTDumper (Stream &stream) :
        m_stream(stream),
        m_base_indentation(stream.GetIndentLevel()),
        m_max_indentation(10)
    {
    }
    
    // MARK: Utility functions
    
    bool KeepDumping ()
    {
        return (m_stream.GetIndentLevel() - m_base_indentation <= m_max_indentation);
    }
    
    void PushIndent()
    {
        m_stream.IndentMore(1);
    }
    
    void PopIndent()
    {
        m_stream.IndentLess(1);
    }
    
    bool Visiting (const void *entity)
    {
        return m_visited_entities.count(entity);
    }
    
    void WillVisit (const void *entity)
    {
        m_visited_entities.insert(entity);
    }
    
    void DidVisit (const void *entity)
    {
        m_visited_entities.erase(entity);
    }
    
public:
    // MARK: DeclVisitor
    
    void VisitDecl                  (clang::Decl *decl);
    void VisitTranslationUnitDecl   (clang::TranslationUnitDecl *translation_unit_decl);
    void VisitNamedDecl             (clang::NamedDecl *named_decl);
    void VisitNamespaceDecl         (clang::NamespaceDecl *namespace_decl);
    void VisitValueDecl             (clang::ValueDecl *value_decl);
    void VisitDeclaratorDecl        (clang::DeclaratorDecl *declarator_decl);
    void VisitVarDecl               (clang::VarDecl *var_decl);
    void VisitTypeDecl              (clang::TypeDecl *type_decl);
    void VisitTagDecl               (clang::TagDecl *tag_decl);
    void VisitRecordDecl            (clang::RecordDecl *record_decl);
    void VisitCXXRecordDecl         (clang::CXXRecordDecl *cxx_record_decl);
    
    // MARK: StmtVisitor
    
    // MARK: TypeVisitor
    
    void VisitType                  (const clang::Type *type);
    void VisitReferenceType         (const clang::ReferenceType *reference_type);
    void VisitLValueReferenceType   (const clang::LValueReferenceType *lvalue_reference_type);
    void VisitPointerType           (const clang::PointerType *pointer_type);
    void VisitTagType               (const clang::TagType *tag_type);
    void VisitRecordType            (const clang::RecordType *record_type);

private:
    llvm::DenseSet <const void *>   m_visited_entities; ///< A set of all entities that have already been printed, to prevent loops
    Stream                         &m_stream;           ///< A stream to print output to
    unsigned                        m_base_indentation; ///< The indentation of m_stream when the ASTDumper was entered
    unsigned                        m_max_indentation;  ///< The maximum depth of indentation (added to m_base_indentation)
public:
    //------------------------------------------------------------------
    /// DumpDecl - Create an ASTDumper and use it to dump a Decl.
    ///
    /// @param[in] stream
    ///     The stream to use when printing output.
    ///
    /// @param[in] decl
    ///     The AST Decl to print.
    //------------------------------------------------------------------
    static void DumpDecl (Stream &stream, clang::Decl *decl)
    {
        ASTDumper dumper(stream);
     
        stream.Printf("---\n");
        
        dumper.::clang::DeclVisitor<ASTDumper, void>::Visit(decl);
    }
    
    //------------------------------------------------------------------
    /// DumpDecl - Create an ASTDumper and use it to dump a Stmt.
    ///
    /// @param[in] stream
    ///     The stream to use when printing output.
    ///
    /// @param[in] stmt
    ///     The AST Stmt to print.
    //------------------------------------------------------------------
    static void DumpStmt (Stream &stream, clang::Stmt *stmt)
    {
        ASTDumper dumper(stream);
        
        stream.Printf("---\n");
        
        dumper.::clang::StmtVisitor<ASTDumper, void>::Visit(stmt);
    }
    
    //------------------------------------------------------------------
    /// DumpDecl - Create an ASTDumper and use it to dump a Type.
    ///
    /// @param[in] stream
    ///     The stream to use when printing output.
    ///
    /// @param[in] type
    ///     The AST Type to print.
    //------------------------------------------------------------------
    static void DumpType (Stream &stream, clang::Type *type)
    {
        ASTDumper dumper(stream);
        
        stream.Printf("---\n");
        
        dumper.::clang::TypeVisitor<ASTDumper, void>::Visit(type);
    }
};

} // namespace lldb_private
