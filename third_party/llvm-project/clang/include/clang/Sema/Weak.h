//===-- UnresolvedSet.h - Unresolved sets of declarations  ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file defines the WeakInfo class, which is used to store
//  information about the target of a #pragma weak directive.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_SEMA_WEAK_H
#define LLVM_CLANG_SEMA_WEAK_H

#include "clang/Basic/SourceLocation.h"

namespace clang {

class IdentifierInfo;

/// Captures information about a \#pragma weak directive.
class WeakInfo {
  IdentifierInfo *alias;  // alias (optional)
  SourceLocation loc;     // for diagnostics
  bool used;              // identifier later declared?
public:
  WeakInfo()
    : alias(nullptr), loc(SourceLocation()), used(false) {}
  WeakInfo(IdentifierInfo *Alias, SourceLocation Loc)
    : alias(Alias), loc(Loc), used(false) {}
  inline IdentifierInfo * getAlias() const { return alias; }
  inline SourceLocation getLocation() const { return loc; }
  void setUsed(bool Used=true) { used = Used; }
  inline bool getUsed() { return used; }
  bool operator==(WeakInfo RHS) const {
    return alias == RHS.getAlias() && loc == RHS.getLocation();
  }
  bool operator!=(WeakInfo RHS) const { return !(*this == RHS); }
};

} // end namespace clang

#endif // LLVM_CLANG_SEMA_WEAK_H
