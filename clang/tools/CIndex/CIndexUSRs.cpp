//===- CIndexUSR.cpp - Clang-C Source Indexing Library --------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the generation and use of USRs from CXEntities.
//
//===----------------------------------------------------------------------===//

#include "CIndexer.h"

extern "C" {

// Some notes on CXEntity:
//
// - Since the 'ordinary' namespace includes functions, data, typedefs,
// ObjC interfaces, thecurrent algorithm is a bit naive (resulting in one
// entity for 2 different types). For example:
//
// module1.m: @interface Foo @end Foo *x;
// module2.m: void Foo(int);
//
// - Since the unique name spans translation units, static data/functions
// within a CXTranslationUnit are *not* currently represented by entities.
// As a result, there will be no entity for the following:
//
// module.m: static void Foo() { }
//

const char *clang_getDeclarationName(CXEntity) {
  return "";
}

const char *clang_getUSR(CXEntity) {
  return "";
}

CXEntity clang_getEntity(const char *URI) {
  return 0;
}

} // end extern "C"
