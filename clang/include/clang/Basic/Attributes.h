//===--- Attributes.h - Attributes header -----------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_BASIC_ATTRIBUTES_H
#define LLVM_CLANG_BASIC_ATTRIBUTES_H

#include "llvm/ADT/Triple.h"
#include "clang/Basic/LangOptions.h"

namespace clang {

class IdentifierInfo;

enum class AttrSyntax {
  /// Is the attribute identifier generally known for any syntax?
  Generic,
  /// Is the identifier known as a GNU-style attribute?
  GNU,
  /// Is the identifier known as a __declspec-style attribute?
  Declspec,
  // Is the identifier known as a C++-style attribute?
  CXX,
  // Is the identifier known as a pragma attribute?
  Pragma
};

/// \brief Return true if we recognize and implement the attribute specified by
/// the given information.
bool hasAttribute(AttrSyntax Syntax, const IdentifierInfo *Scope,
                  const IdentifierInfo *Attr, const llvm::Triple &T,
                  const LangOptions &LangOpts);

} // end namespace clang

#endif // LLVM_CLANG_BASIC_ATTRIBUTES_H
