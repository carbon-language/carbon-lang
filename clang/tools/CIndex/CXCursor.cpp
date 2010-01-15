//===- CXCursor.cpp - Routines for manipulating CXCursors -----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines routines for manipulating CXCursors.
//
//===----------------------------------------------------------------------===//

#include "CXCursor.h"
#include "clang/AST/Decl.h"

using namespace clang;

CXCursor cxcursor::MakeCXCursor(CXCursorKind K, Decl *D) {
  CXCursor C = { K, D, 0, 0 };
  return C;  
}

CXCursor cxcursor::MakeCXCursor(CXCursorKind K, Decl *D, Stmt *S) {
  assert(clang_isReference(K));
  CXCursor C = { K, D, S, 0 };
  return C;  
}

