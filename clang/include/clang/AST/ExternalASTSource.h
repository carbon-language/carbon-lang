//===--- ExternalASTSource.h - Abstract External AST Interface --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the ExternalASTSource interface, which enables
//  construction of AST nodes from some external source.x
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_CLANG_AST_EXTERNAL_AST_SOURCE_H
#define LLVM_CLANG_AST_EXTERNAL_AST_SOURCE_H

#include "clang/AST/DeclarationName.h"
#include "clang/AST/Type.h"
#include "llvm/ADT/SmallVector.h"
#include <cassert>
#include <vector>
namespace clang {

class ASTConsumer;
class Decl;
class DeclContext;
class ExternalSemaSource; // layering violation required for downcasting
class Stmt;

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
  /// \brief Whether this AST source also provides information for
  /// semantic analysis.
  bool SemaSource;

  friend class ExternalSemaSource;

public:
  ExternalASTSource() : SemaSource(false) { }

  virtual ~ExternalASTSource();

  /// \brief RAII class for safely pairing a StartedDeserializing call
  /// with FinishedDeserializing.
  class Deserializing {
    ExternalASTSource *Source;
  public:
    explicit Deserializing(ExternalASTSource *source) : Source(source) {
      assert(Source);
      Source->StartedDeserializing();
    }
    ~Deserializing() {
      Source->FinishedDeserializing();
    }
  };

  /// \brief Resolve a declaration ID into a declaration, potentially
  /// building a new declaration.
  ///
  /// This method only needs to be implemented if the AST source ever
  /// passes back decl sets as VisibleDeclaration objects.
  virtual Decl *GetExternalDecl(uint32_t ID) = 0;

  /// \brief Resolve a selector ID into a selector.
  ///
  /// This operation only needs to be implemented if the AST source
  /// returns non-zero for GetNumKnownSelectors().
  virtual Selector GetExternalSelector(uint32_t ID) = 0;

  /// \brief Returns the number of selectors known to the external AST
  /// source.
  virtual uint32_t GetNumExternalSelectors() = 0;

  /// \brief Resolve the offset of a statement in the decl stream into
  /// a statement.
  ///
  /// This operation is meant to be used via a LazyOffsetPtr.  It only
  /// needs to be implemented if the AST source uses methods like
  /// FunctionDecl::setLazyBody when building decls.
  virtual Stmt *GetExternalDeclStmt(uint64_t Offset) = 0;

  /// \brief Finds all declarations with the given name in the
  /// given context.
  ///
  /// Generally the final step of this method is either to call
  /// SetExternalVisibleDeclsForName or to recursively call lookup on
  /// the DeclContext after calling SetExternalVisibleDecls.
  virtual DeclContext::lookup_result
  FindExternalVisibleDeclsByName(const DeclContext *DC,
                                 DeclarationName Name) = 0;

  /// \brief Finds all declarations lexically contained within the given
  /// DeclContext.
  ///
  /// \return true if an error occurred
  virtual bool FindExternalLexicalDecls(const DeclContext *DC,
                                llvm::SmallVectorImpl<Decl*> &Result) = 0;

  /// \brief Notify ExternalASTSource that we started deserialization of
  /// a decl or type so until FinishedDeserializing is called there may be
  /// decls that are initializing. Must be paired with FinishedDeserializing.
  ///
  /// The default implementation of this method is a no-op.
  virtual void StartedDeserializing() { }

  /// \brief Notify ExternalASTSource that we finished the deserialization of
  /// a decl or type. Must be paired with StartedDeserializing.
  ///
  /// The default implementation of this method is a no-op.
  virtual void FinishedDeserializing() { }

  /// \brief Function that will be invoked when we begin parsing a new
  /// translation unit involving this external AST source.
  ///
  /// The default implementation of this method is a no-op.
  virtual void StartTranslationUnit(ASTConsumer *Consumer) { }

  /// \brief Print any statistics that have been gathered regarding
  /// the external AST source.
  ///
  /// The default implementation of this method is a no-op.
  virtual void PrintStats();

protected:
  /// \brief Initialize the context's lookup map with the given decls.
  /// It is assumed that none of the declarations are redeclarations of
  /// each other.
  static void SetExternalVisibleDecls(const DeclContext *DC,
                  const llvm::SmallVectorImpl<VisibleDeclaration> &Decls);

  /// \brief Initialize the context's lookup map with the given decls.
  /// It is assumed that none of the declarations are redeclarations of
  /// each other.
  static void SetExternalVisibleDecls(const DeclContext *DC,
                              const llvm::SmallVectorImpl<NamedDecl*> &Decls);

  static DeclContext::lookup_result
  SetExternalVisibleDeclsForName(const DeclContext *DC,
                                 const VisibleDeclaration &VD);

  static DeclContext::lookup_result
  SetExternalVisibleDeclsForName(const DeclContext *DC,
                                 DeclarationName Name,
                                 llvm::SmallVectorImpl<NamedDecl*> &Decls);

  static DeclContext::lookup_result
  SetNoExternalVisibleDeclsForName(const DeclContext *DC,
                                   DeclarationName Name);
};

/// \brief A lazy pointer to an AST node (of base type T) that resides
/// within an external AST source.
///
/// The AST node is identified within the external AST source by a
/// 63-bit offset, and can be retrieved via an operation on the
/// external AST source itself.
template<typename T, T* (ExternalASTSource::*Get)(uint64_t Offset)>
struct LazyOffsetPtr {
  /// \brief Either a pointer to an AST node or the offset within the
  /// external AST source where the AST node can be found.
  ///
  /// If the low bit is clear, a pointer to the AST node. If the low
  /// bit is set, the upper 63 bits are the offset.
  mutable uint64_t Ptr;

public:
  LazyOffsetPtr() : Ptr(0) { }

  explicit LazyOffsetPtr(T *Ptr) : Ptr(reinterpret_cast<uint64_t>(Ptr)) { }
  explicit LazyOffsetPtr(uint64_t Offset) : Ptr((Offset << 1) | 0x01) {
    assert((Offset << 1 >> 1) == Offset && "Offsets must require < 63 bits");
    if (Offset == 0)
      Ptr = 0;
  }

  LazyOffsetPtr &operator=(T *Ptr) {
    this->Ptr = reinterpret_cast<uint64_t>(Ptr);
    return *this;
  }

  LazyOffsetPtr &operator=(uint64_t Offset) {
    assert((Offset << 1 >> 1) == Offset && "Offsets must require < 63 bits");
    if (Offset == 0)
      Ptr = 0;
    else
      Ptr = (Offset << 1) | 0x01;

    return *this;
  }

  /// \brief Whether this pointer is non-NULL.
  ///
  /// This operation does not require the AST node to be deserialized.
  operator bool() const { return Ptr != 0; }

  /// \brief Whether this pointer is currently stored as an offset.
  bool isOffset() const { return Ptr & 0x01; }

  /// \brief Retrieve the pointer to the AST node that this lazy pointer
  ///
  /// \param Source the external AST source.
  ///
  /// \returns a pointer to the AST node.
  T* get(ExternalASTSource *Source) const {
    if (isOffset()) {
      assert(Source &&
             "Cannot deserialize a lazy pointer without an AST source");
      Ptr = reinterpret_cast<uint64_t>((Source->*Get)(Ptr >> 1));
    }
    return reinterpret_cast<T*>(Ptr);
  }
};

/// \brief A lazy pointer to a statement.
typedef LazyOffsetPtr<Stmt, &ExternalASTSource::GetExternalDeclStmt>
  LazyDeclStmtPtr;

} // end namespace clang

#endif // LLVM_CLANG_AST_EXTERNAL_AST_SOURCE_H
