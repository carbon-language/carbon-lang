//=== MangleNumberingContext.h - Context for mangling numbers ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the LambdaBlockMangleContext interface, which keeps track
//  of the Itanium C++ ABI mangling numbers for lambda expressions and block
//  literals.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_CLANG_MANGLENUMBERINGCONTEXT_H
#define LLVM_CLANG_MANGLENUMBERINGCONTEXT_H

#include "clang/Basic/LLVM.h"
#include "clang/Sema/Scope.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"

namespace clang {

class BlockDecl;
class CXXMethodDecl;
class IdentifierInfo;
class TagDecl;
class Type;
class VarDecl;

/// \brief Keeps track of the mangled names of lambda expressions and block
/// literals within a particular context.
class MangleNumberingContext 
    : public RefCountedBase<MangleNumberingContext> {
  llvm::DenseMap<const Type *, unsigned> ManglingNumbers;

public:
  virtual ~MangleNumberingContext() {}

  /// \brief Retrieve the mangling number of a new lambda expression with the
  /// given call operator within this context.
  unsigned getManglingNumber(const CXXMethodDecl *CallOperator);

  /// \brief Retrieve the mangling number of a new block literal within this
  /// context.
  unsigned getManglingNumber(const BlockDecl *BD);

  /// Static locals are numbered by source order.
  unsigned getStaticLocalNumber(const VarDecl *VD);

  /// \brief Retrieve the mangling number of a static local variable within
  /// this context.
  virtual unsigned getManglingNumber(const VarDecl *VD, Scope *S) = 0;

  /// \brief Retrieve the mangling number of a static local variable within
  /// this context.
  virtual unsigned getManglingNumber(const TagDecl *TD, Scope *S) = 0;
};
  
} // end namespace clang
#endif
