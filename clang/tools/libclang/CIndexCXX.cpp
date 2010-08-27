//===- CIndexCXX.cpp - Clang-C Source Indexing Library --------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the libclang support for C++ cursors.
//
//===----------------------------------------------------------------------===//

#include "CIndexer.h"
#include "CXCursor.h"
#include "CXType.h"
#include "clang/AST/DeclCXX.h"

using namespace clang;
using namespace clang::cxstring;

extern "C" {

unsigned clang_isVirtualBase(CXCursor C) {
  if (C.kind != CXCursor_CXXBaseSpecifier)
    return 0;
  
  CXXBaseSpecifier *B = cxcursor::getCursorCXXBaseSpecifier(C);
  return B->isVirtual();
}

enum CX_CXXAccessSpecifier clang_getCXXAccessSpecifier(CXCursor C) {
  if (C.kind != CXCursor_CXXBaseSpecifier)
    return CX_CXXInvalidAccessSpecifier;
  
  CXXBaseSpecifier *B = cxcursor::getCursorCXXBaseSpecifier(C);
  switch (B->getAccessSpecifier()) {
    case AS_public: return CX_CXXPublic;
    case AS_protected: return CX_CXXProtected;
    case AS_private: return CX_CXXPrivate;
    case AS_none: return CX_CXXInvalidAccessSpecifier;
  }
  
  // FIXME: Clang currently thinks this is reachable.
  return CX_CXXInvalidAccessSpecifier;
}

} // end extern "C"
