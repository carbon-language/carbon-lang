//===- CXCursor.h - Routines for manipulating CXCursors -------------------===//
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

#ifndef LLVM_CLANG_CXCURSOR_H
#define LLVM_CLANG_CXCursor_H

#include "clang-c/Index.h"

namespace clang {

class Decl;
class Expr;
class NamedDecl;
class Stmt;

namespace cxcursor {
  
CXCursor MakeCXCursor(CXCursorKind K, clang::Decl *D);  
CXCursor MakeCXCursor(CXCursorKind K, clang::Decl *D, clang::Stmt *S);
CXCursor MakeCXCursor(clang::Decl *D);

Decl *getCursorDecl(CXCursor Cursor);
Expr *getCursorExpr(CXCursor Cursor);
Stmt *getCursorStmt(CXCursor Cursor);
Decl *getCursorReferringDecl(CXCursor Cursor);
NamedDecl *getCursorInterfaceParent(CXCursor Cursor);
  
bool operator==(CXCursor X, CXCursor Y);
  
inline bool operator!=(CXCursor X, CXCursor Y) {
  return !(X == Y);
}

}} // end namespace: clang::cxcursor

#endif
