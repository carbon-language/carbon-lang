//== CheckerHelpers.h - Helper functions for checkers ------------*- C++ -*--=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines CheckerVisitor.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_CHECKER_PATHSENSITIVE_CHECKERHELPERS
#define LLVM_CLANG_CHECKER_PATHSENSITIVE_CHECKERHELPERS

#include "clang/AST/Stmt.h"

namespace clang {

bool containsMacro(const Stmt *S);
bool containsEnum(const Stmt *S);
bool containsStaticLocal(const Stmt *S);
bool containsBuiltinOffsetOf(const Stmt *S);
template <class T> bool containsStmt(const Stmt *S) {
  if (isa<T>(S))
      return true;

  for (Stmt::const_child_iterator I = S->child_begin(); I != S->child_end();
      ++I)
    if (const Stmt *child = *I)
      if (containsStmt<T>(child))
        return true;

  return false;
}

}

#endif
