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

#include "llvm/Support/DataTypes.h"
#include <cassert>

namespace llvm {
  class raw_ostream;
}

namespace clang {

class ASTContext;
class DeclContext;
class Type;

/// \brief Represents a single component in a C++ nested-name-specifier.
///
/// C++ nested-name-specifiers are the prefixes to qualified
/// namespaces. For example, "foo::" in "foo::x" is a
/// nested-name-specifier. Multiple nested-name-specifiers can be
/// strung together to build qualified names, e.g., "foo::bar::" in
/// "foo::bar::x". Each NestedNameSpecifier class contains one of the
/// terms, e.g., "foo::" or "bar::", which may be represented either
/// as a type or as a DeclContext.
class NestedNameSpecifier {
  /// \brief A DeclContext or Type pointer, depending on whether the
  /// low bit is set.
  uintptr_t Data;

public:
  NestedNameSpecifier() : Data(0) { }

  /// \brief Construct a nested name specifier that refers to a type.
  NestedNameSpecifier(const Type *T) { 
    Data = reinterpret_cast<uintptr_t>(T);
    assert((Data & 0x01) == 0 && "cv-qualified type in nested-name-specifier");
    Data |= 0x01;
  }

  /// \brief Construct nested name specifier that refers to a
  /// DeclContext.
  NestedNameSpecifier(const DeclContext *DC) {
    Data = reinterpret_cast<uintptr_t>(DC);
    assert((Data & 0x01) == 0 && "Badly aligned DeclContext pointer");
  }

  /// \brief Determines whether this nested-name-specifier refers to a
  /// type. Otherwise, it refers to a DeclContext.
  bool isType() const { return Data & 0x01; }

  /// \brief Compute the declaration context to which this
  /// nested-name-specifier refers.
  ///
  /// This routine computes the declaration context referenced by this
  /// nested-name-specifier. The nested-name-specifier may store
  /// either a DeclContext (the trivial case) or a non-dependent type
  /// (which will have an associated DeclContext). It is an error to
  /// invoke this routine when the nested-name-specifier refers to a
  /// dependent type.
  ///
  /// \returns The stored DeclContext, if the nested-name-specifier
  /// stores a DeclContext. If the nested-name-specifier stores a
  /// non-dependent type, returns the DeclContext associated with that
  /// type.
  DeclContext *computeDeclContext(ASTContext &Context) const;

  /// \brief Retrieve the nested-name-specifier as a type. 
  ///
  /// \returns The stored type. If the nested-name-specifier does not
  /// store a type, returns NULL.
  Type *getAsType() const {
    if (Data & 0x01)
      return reinterpret_cast<Type *>(Data & ~0x01);

    return 0;
  }

  /// \brief Retrieves the nested-name-specifier as a DeclContext.
  ///
  /// \returns The stored DeclContext. If the nested-name-specifier
  /// does not store a DeclContext, returns NULL.
  DeclContext *getAsDeclContext() const {
    if (Data & 0x01)
      return 0;
    return reinterpret_cast<DeclContext *>(Data);
  }

  /// \brief Retrieve nested name specifier as an opaque pointer.
  void *getAsOpaquePtr() const { return reinterpret_cast<void *>(Data); }

  /// \brief Reconstruct a nested name specifier from an opaque pointer.
  static NestedNameSpecifier getFromOpaquePtr(void *Ptr) {
    NestedNameSpecifier NS;
    NS.Data = reinterpret_cast<uintptr_t>(Ptr);
    return NS;
  }

  static void Print(llvm::raw_ostream &OS, const NestedNameSpecifier *First,
                    const NestedNameSpecifier *Last);
};

}

#endif
