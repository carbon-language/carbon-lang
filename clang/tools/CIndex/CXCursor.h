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
class Stmt;

namespace cxcursor {
  
CXCursor MakeCXCursor(CXCursorKind K, clang::Decl *D);  
CXCursor MakeCXCursor(CXCursorKind K, clang::Decl *D, clang::Stmt *S);

}} // end namespace: clang::cxcursor

#endif
