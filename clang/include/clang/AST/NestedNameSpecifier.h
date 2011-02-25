//===--- NestedNameSpecifier.h - C++ nested name specifiers -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the NestedNameSpecifier class, which represents
//  a C++ nested-name-specifier.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_CLANG_AST_NESTEDNAMESPECIFIER_H
#define LLVM_CLANG_AST_NESTEDNAMESPECIFIER_H

#include "clang/Basic/Diagnostic.h"
#include "llvm/ADT/FoldingSet.h"
#include "llvm/ADT/PointerIntPair.h"

namespace llvm {
  class raw_ostream;
}

namespace clang {

class ASTContext;
class NamespaceAliasDecl;
class NamespaceDecl;
class IdentifierInfo;
struct PrintingPolicy;
class Type;
class TypeLoc;
class LangOptions;

/// \brief Represents a C++ nested name specifier, such as
/// "::std::vector<int>::".
///
/// C++ nested name specifiers are the prefixes to qualified
/// namespaces. For example, "foo::" in "foo::x" is a nested name
/// specifier. Nested name specifiers are made up of a sequence of
/// specifiers, each of which can be a namespace, type, identifier
/// (for dependent names), or the global specifier ('::', must be the
/// first specifier).
class NestedNameSpecifier : public llvm::FoldingSetNode {

  /// \brief Enumeration describing
  enum StoredSpecifierKind {
    StoredIdentifier = 0,
    StoredNamespaceOrAlias = 1,
    StoredTypeSpec = 2,
    StoredTypeSpecWithTemplate = 3
  };

  /// \brief The nested name specifier that precedes this nested name
  /// specifier.
  ///
  /// The pointer is the nested-name-specifier that precedes this
  /// one. The integer stores one of the first four values of type
  /// SpecifierKind.
  llvm::PointerIntPair<NestedNameSpecifier *, 2, StoredSpecifierKind> Prefix;

  /// \brief The last component in the nested name specifier, which
  /// can be an identifier, a declaration, or a type.
  ///
  /// When the pointer is NULL, this specifier represents the global
  /// specifier '::'. Otherwise, the pointer is one of
  /// IdentifierInfo*, Namespace*, or Type*, depending on the kind of
  /// specifier as encoded within the prefix.
  void* Specifier;

public:
  /// \brief The kind of specifier that completes this nested name
  /// specifier.
  enum SpecifierKind {
    /// \brief An identifier, stored as an IdentifierInfo*.
    Identifier,
    /// \brief A namespace, stored as a NamespaceDecl*.
    Namespace,
    /// \brief A namespace alias, stored as a NamespaceAliasDecl*.
    NamespaceAlias,
    /// \brief A type, stored as a Type*.
    TypeSpec,
    /// \brief A type that was preceded by the 'template' keyword,
    /// stored as a Type*.
    TypeSpecWithTemplate,
    /// \brief The global specifier '::'. There is no stored value.
    Global
  };

private:
  /// \brief Builds the global specifier.
  NestedNameSpecifier() : Prefix(0, StoredIdentifier), Specifier(0) { }

  /// \brief Copy constructor used internally to clone nested name
  /// specifiers.
  NestedNameSpecifier(const NestedNameSpecifier &Other)
    : llvm::FoldingSetNode(Other), Prefix(Other.Prefix),
      Specifier(Other.Specifier) {
  }

  NestedNameSpecifier &operator=(const NestedNameSpecifier &); // do not implement

  /// \brief Either find or insert the given nested name specifier
  /// mockup in the given context.
  static NestedNameSpecifier *FindOrInsert(const ASTContext &Context,
                                           const NestedNameSpecifier &Mockup);

public:
  /// \brief Builds a specifier combining a prefix and an identifier.
  ///
  /// The prefix must be dependent, since nested name specifiers
  /// referencing an identifier are only permitted when the identifier
  /// cannot be resolved.
  static NestedNameSpecifier *Create(const ASTContext &Context,
                                     NestedNameSpecifier *Prefix,
                                     IdentifierInfo *II);

  /// \brief Builds a nested name specifier that names a namespace.
  static NestedNameSpecifier *Create(const ASTContext &Context,
                                     NestedNameSpecifier *Prefix,
                                     NamespaceDecl *NS);

  /// \brief Builds a nested name specifier that names a namespace alias.
  static NestedNameSpecifier *Create(const ASTContext &Context,
                                     NestedNameSpecifier *Prefix,
                                     NamespaceAliasDecl *Alias);

  /// \brief Builds a nested name specifier that names a type.
  static NestedNameSpecifier *Create(const ASTContext &Context,
                                     NestedNameSpecifier *Prefix,
                                     bool Template, const Type *T);

  /// \brief Builds a specifier that consists of just an identifier.
  ///
  /// The nested-name-specifier is assumed to be dependent, but has no
  /// prefix because the prefix is implied by something outside of the
  /// nested name specifier, e.g., in "x->Base::f", the "x" has a dependent
  /// type.
  static NestedNameSpecifier *Create(const ASTContext &Context,
                                     IdentifierInfo *II);

  /// \brief Returns the nested name specifier representing the global
  /// scope.
  static NestedNameSpecifier *GlobalSpecifier(const ASTContext &Context);

  /// \brief Return the prefix of this nested name specifier.
  ///
  /// The prefix contains all of the parts of the nested name
  /// specifier that preced this current specifier. For example, for a
  /// nested name specifier that represents "foo::bar::", the current
  /// specifier will contain "bar::" and the prefix will contain
  /// "foo::".
  NestedNameSpecifier *getPrefix() const { return Prefix.getPointer(); }

  /// \brief Determine what kind of nested name specifier is stored.
  SpecifierKind getKind() const;

  /// \brief Retrieve the identifier stored in this nested name
  /// specifier.
  IdentifierInfo *getAsIdentifier() const {
    if (Prefix.getInt() == StoredIdentifier)
      return (IdentifierInfo *)Specifier;

    return 0;
  }

  /// \brief Retrieve the namespace stored in this nested name
  /// specifier.
  NamespaceDecl *getAsNamespace() const;

  /// \brief Retrieve the namespace alias stored in this nested name
  /// specifier.
  NamespaceAliasDecl *getAsNamespaceAlias() const;

  /// \brief Retrieve the type stored in this nested name specifier.
  const Type *getAsType() const {
    if (Prefix.getInt() == StoredTypeSpec ||
        Prefix.getInt() == StoredTypeSpecWithTemplate)
      return (const Type *)Specifier;

    return 0;
  }

  /// \brief Whether this nested name specifier refers to a dependent
  /// type or not.
  bool isDependent() const;

  /// \brief Whether this nested-name-specifier contains an unexpanded
  /// parameter pack (for C++0x variadic templates).
  bool containsUnexpandedParameterPack() const;

  /// \brief Print this nested name specifier to the given output
  /// stream.
  void print(llvm::raw_ostream &OS, const PrintingPolicy &Policy) const;

  void Profile(llvm::FoldingSetNodeID &ID) const {
    ID.AddPointer(Prefix.getOpaqueValue());
    ID.AddPointer(Specifier);
  }

  /// \brief Dump the nested name specifier to standard output to aid
  /// in debugging.
  void dump(const LangOptions &LO);
};

/// \brief A C++ nested-name-specifier augmented with source location
/// information.
class NestedNameSpecifierLoc {
  NestedNameSpecifier *Qualifier;
  void *Data;

  /// \brief Determines the data length for the last component in the
  /// given nested-name-specifier.
  static unsigned getLocalDataLength(NestedNameSpecifier *Qualifier);

  /// \brief Determines the data length for the entire
  /// nested-name-specifier.
  static unsigned getDataLength(NestedNameSpecifier *Qualifier);

public:
  /// \brief Construct an empty nested-name-specifier.
  NestedNameSpecifierLoc() : Qualifier(0), Data(0) { }
  
  /// \brief Construct a nested-name-specifier with source location information
  /// from 
  NestedNameSpecifierLoc(NestedNameSpecifier *Qualifier, void *Data)
    : Qualifier(Qualifier), Data(Data) { }
  
  /// \brief Evalutes true when this nested-name-specifier location is
  /// non-empty.
  operator bool() const { return Qualifier; }

  /// \brief Retrieve the nested-name-specifier to which this instance
  /// refers.
  NestedNameSpecifier *getNestedNameSpecifier() const {
    return Qualifier;
  }

  /// \brief Retrieve the opaque pointer that refers to source-location data.
  void *getOpaqueData() const { return Data; }
  
  /// \brief Retrieve the source range covering the entirety of this
  /// nested-name-specifier.
  ///
  /// For example, if this instance refers to a nested-name-specifier
  /// \c ::std::vector<int>::, the returned source range would cover
  /// from the initial '::' to the last '::'.
  SourceRange getSourceRange() const;

  /// \brief Retrieve the source range covering just the last part of
  /// this nested-name-specifier, not including the prefix.
  ///
  /// For example, if this instance refers to a nested-name-specifier
  /// \c ::std::vector<int>::, the returned source range would cover
  /// from "vector" to the last '::'.
  SourceRange getLocalSourceRange() const;

  /// \brief Retrieve the location of the beginning of this
  /// nested-name-specifier.
  SourceLocation getBeginLoc() const { 
    return getSourceRange().getBegin();
  }

  /// \brief Retrieve the location of the end of this
  /// nested-name-specifier.
  SourceLocation getEndLoc() const { 
    return getSourceRange().getEnd();
  }

  /// \brief Retrieve the location of the beginning of this
  /// component of the nested-name-specifier.
  SourceLocation getLocalBeginLoc() const { 
    return getLocalSourceRange().getBegin();
  }
  
  /// \brief Retrieve the location of the end of this component of the
  /// nested-name-specifier.
  SourceLocation getLocalEndLoc() const { 
    return getLocalSourceRange().getEnd();
  }

  /// \brief Return the prefix of this nested-name-specifier.
  ///
  /// For example, if this instance refers to a nested-name-specifier
  /// \c ::std::vector<int>::, the prefix is \c ::std::. Note that the
  /// returned prefix may be empty, if this is the first component of
  /// the nested-name-specifier.
  NestedNameSpecifierLoc getPrefix() const {
    if (!Qualifier)
      return *this;

    return NestedNameSpecifierLoc(Qualifier->getPrefix(), Data);
  }

  /// \brief For a nested-name-specifier that refers to a type,
  /// retrieve the type with source-location information.
  TypeLoc getTypeLoc() const;

  /// \brief Determines the data length for the entire
  /// nested-name-specifier.
  unsigned getDataLength() const { return getDataLength(Qualifier); }
  
  friend bool operator==(NestedNameSpecifierLoc X, 
                         NestedNameSpecifierLoc Y) {
    return X.Qualifier == Y.Qualifier && X.Data == Y.Data;
  }

  friend bool operator!=(NestedNameSpecifierLoc X, 
                         NestedNameSpecifierLoc Y) {
    return !(X == Y);
  }
};

/// Insertion operator for diagnostics.  This allows sending NestedNameSpecifiers
/// into a diagnostic with <<.
inline const DiagnosticBuilder &operator<<(const DiagnosticBuilder &DB,
                                           NestedNameSpecifier *NNS) {
  DB.AddTaggedVal(reinterpret_cast<intptr_t>(NNS),
                  Diagnostic::ak_nestednamespec);
  return DB;
}

}

#endif
