//===--- LoopHint.h - Types for LoopHint ------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_SEMA_LOOPHINT_H
#define LLVM_CLANG_SEMA_LOOPHINT_H

#include "clang/Basic/IdentifierTable.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Sema/AttributeList.h"
#include "clang/Sema/Ownership.h"

namespace clang {

/// \brief Loop hint specified by a pragma loop directive.
struct LoopHint {
  SourceRange Range;
  Expr *ValueExpr;
  IdentifierLoc *LoopLoc;
  IdentifierLoc *ValueLoc;
  IdentifierLoc *OptionLoc;
};

} // end namespace clang

#endif // LLVM_CLANG_SEMA_LOOPHINT_H
