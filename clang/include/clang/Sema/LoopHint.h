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

/// \brief Loop optimization hint for loop and unroll pragmas.
struct LoopHint {
  // Source range of the directive.
  SourceRange Range;
  // Identifier corresponding to the name of the pragma.  "loop" for
  // "#pragma clang loop" directives and "unroll" for "#pragma unroll"
  // hints.
  IdentifierLoc *PragmaNameLoc;
  // Name of the loop hint.  Examples: "unroll", "vectorize".  In the
  // "#pragma unroll" case, this is identical to PragmaNameLoc.
  IdentifierLoc *OptionLoc;
  // Identifier for the hint argument.  If null, then the hint has no argument
  // such as for "#pragma unroll".
  IdentifierLoc *ValueLoc;
  // Expression for the hint argument if it exists, null otherwise.
  Expr *ValueExpr;
};

} // end namespace clang

#endif // LLVM_CLANG_SEMA_LOOPHINT_H
