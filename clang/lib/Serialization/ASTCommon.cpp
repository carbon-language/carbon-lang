//===--- ASTCommon.cpp - Common stuff for ASTReader/ASTWriter----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines common functions that both ASTReader and ASTWriter use.
//
//===----------------------------------------------------------------------===//

#include "ASTCommon.h"
#include "clang/Basic/IdentifierTable.h"
#include "llvm/ADT/StringExtras.h"

using namespace clang;

unsigned serialization::ComputeHash(Selector Sel) {
  unsigned N = Sel.getNumArgs();
  if (N == 0)
    ++N;
  unsigned R = 5381;
  for (unsigned I = 0; I != N; ++I)
    if (IdentifierInfo *II = Sel.getIdentifierInfoForSlot(I))
      R = llvm::HashString(II->getName(), R);
  return R;
}
