//===--- ExternalASTSource.h - Abstract External AST Interface --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the ExternalASTSource interface, 
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_CLANG_AST_EXTERNAL_AST_SOURCE_H
#define LLVM_CLANG_AST_EXTERNAL_AST_SOURCE_H

#include "clang/AST/DeclarationName.h"
#include "clang/AST/Type.h"
#include "llvm/ADT/SmallVector.h"
namespace clang {

class Decl;
class DeclContext;

/// \brief The deserialized representation of a set of declarations
/// with the same name that are visible in a given context.
struct VisibleDeclaration {
  /// \brief The name of the declarations.
  DeclarationName Name;

  /// \brief The ID numbers of all of the declarations with this name. 
  ///
  /// These declarations have not necessarily been de-serialized.
  llvm::SmallVector<unsigned, 4> Declarations;
};

/// \brief Abstract interface for external sources of AST nodes.
///
/// External AST sources provide AST nodes constructed from some
/// external source, such as a precompiled header. External AST
/// sources can resolve types and declarations from abstract IDs into
/// actual type and declaration nodes, and read parts of declaration
/// contexts.
class ExternalASTSource {
public:
  virtual ~ExternalASTSource();

  /// \brief Resolve a type ID into a type, potentially building a new
  /// type.
  virtual QualType GetType(unsigned ID) = 0;

  /// \brief Resolve a declaration ID into a declaration, potentially
  /// building a new declaration.
  virtual Decl *GetDecl(unsigned ID) = 0;

  /// \brief Read all of the declarations lexically stored in a
  /// declaration context.
  ///
  /// \param DC The declaration context whose declarations will be
  /// read.
  ///
  /// \param Decls Vector that will contain the declarations loaded
  /// from the external source. The caller is responsible for merging
  /// these declarations with any declarations already stored in the
  /// declaration context.
  ///
  /// \returns true if there was an error while reading the
  /// declarations for this declaration context.
  virtual bool ReadDeclsLexicallyInContext(DeclContext *DC,
                                  llvm::SmallVectorImpl<unsigned> &Decls) = 0;

  /// \brief Read all of the declarations visible from a declaration
  /// context.
  ///
  /// \param DC The declaration context whose visible declarations
  /// will be read.
  ///
  /// \param Decls A vector of visible declaration structures,
  /// providing the mapping from each name visible in the declaration
  /// context to the declaration IDs of declarations with that name.
  ///
  /// \returns true if there was an error while reading the
  /// declarations for this declaration context.
  virtual bool ReadDeclsVisibleInContext(DeclContext *DC,
                       llvm::SmallVectorImpl<VisibleDeclaration> & Decls) = 0;

  /// \brief Print any statistics that have been gathered regarding
  /// the external AST source.
  virtual void PrintStats();
};

} // end namespace clang

#endif // LLVM_CLANG_AST_EXTERNAL_AST_SOURCE_H
