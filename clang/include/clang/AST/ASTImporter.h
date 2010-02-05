//===--- ASTImporter.h - Importing ASTs from other Contexts -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the ASTImporter class which imports AST nodes from one
//  context into another context.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_CLANG_AST_ASTIM PORTER_H
#define LLVM_CLANG_AST_ASTIMPORTER_H

#include "clang/AST/Type.h"
#include "clang/AST/DeclarationName.h"
#include "clang/Basic/SourceLocation.h"

namespace clang {
  class ASTContext;
  class Decl;
  class DeclContext;
  class Diagnostic;
  class Expr;
  class IdentifierInfo;
  class NestedNameSpecifier;
  class Stmt;
  class TypeSourceInfo;
  
  /// \brief Imports selected nodes from one AST context into another context,
  /// merging AST nodes where appropriate.
  class ASTImporter {
    /// \brief The contexts we're importing to and from.
    ASTContext &ToContext, &FromContext;
    
    /// \brief The diagnostics object that we should use to emit diagnostics
    /// within the context we're importing to and from.
    Diagnostic &ToDiags, &FromDiags;
          
  public:
    ASTImporter(ASTContext &ToContext, Diagnostic &ToDiags,
                ASTContext &FromContext, Diagnostic &FromDiags);
    
    ~ASTImporter();
    
    /// \brief Import the given type from the "from" context into the "to"
    /// context.
    ///
    /// \returns the equivalent type in the "to" context, or a NULL type if
    /// an error occurred.
    QualType Import(QualType FromT);

    /// \brief Import the given type source information from the
    /// "from" context into the "to" context.
    ///
    /// \returns the equivalent type source information in the "to"
    /// context, or NULL if an error occurred.
    TypeSourceInfo *Import(TypeSourceInfo *FromTSI);

    /// \brief Import the given declaration from the "from" context into the 
    /// "to" context.
    ///
    /// \returns the equivalent declaration in the "to" context, or a NULL type 
    /// if an error occurred.
    Decl *Import(Decl *FromD);

    /// \brief Import the given declaration context from the "from"
    /// AST context into the "to" AST context.
    ///
    /// \returns the equivalent declaration context in the "to"
    /// context, or a NULL type if an error occurred.
    DeclContext *ImportContext(DeclContext *FromDC);
    
    /// \brief Import the given expression from the "from" context into the
    /// "to" context.
    ///
    /// \returns the equivalent expression in the "to" context, or NULL if
    /// an error occurred.
    Expr *Import(Expr *FromE);

    /// \brief Import the given statement from the "from" context into the
    /// "to" context.
    ///
    /// \returns the equivalent statement in the "to" context, or NULL if
    /// an error occurred.
    Stmt *Import(Stmt *FromS);

    /// \brief Import the given nested-name-specifier from the "from"
    /// context into the "to" context.
    ///
    /// \returns the equivalent nested-name-specifier in the "to"
    /// context, or NULL if an error occurred.
    NestedNameSpecifier *Import(NestedNameSpecifier *FromNNS);

    /// \brief Import the given source location from the "from" context into
    /// the "to" context.
    ///
    /// \returns the equivalent source location in the "to" context, or an
    /// invalid source location if an error occurred.
    SourceLocation Import(SourceLocation FromLoc);

    /// \brief Import the given source range from the "from" context into
    /// the "to" context.
    ///
    /// \returns the equivalent source range in the "to" context, or an
    /// invalid source location if an error occurred.
    SourceRange Import(SourceRange FromRange);

    /// \brief Import the given declaration name from the "from"
    /// context into the "to" context.
    ///
    /// \returns the equivalent declaration name in the "to" context,
    /// or an empty declaration name if an error occurred.
    DeclarationName Import(DeclarationName FromName);

    /// \brief Import the given identifier from the "from" context
    /// into the "to" context.
    ///
    /// \returns the equivalent identifier in the "to" context.
    IdentifierInfo *Import(IdentifierInfo *FromId);

    /// \brief Retrieve the context that AST nodes are being imported into.
    ASTContext &getToContext() const { return ToContext; }
    
    /// \brief Retrieve the context that AST nodes are being imported from.
    ASTContext &getFromContext() const { return FromContext; }
    
    /// \brief Retrieve the diagnostics object to use to report errors within
    /// the context we're importing into.
    Diagnostic &getToDiags() const { return ToDiags; }

    /// \brief Retrieve the diagnostics object to use to report errors within
    /// the context we're importing from.
    Diagnostic &getFromDiags() const { return FromDiags; }
  };
}

#endif // LLVM_CLANG_AST_ASTIMPORTER_H
