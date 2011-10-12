//===- CXTranslationUnit.h - Routines for manipulating CXTranslationUnits -===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines routines for manipulating CXTranslationUnits.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_CXTRANSLATIONUNIT_H
#define LLVM_CLANG_CXTRANSLATIONUNIT_H

extern "C" {
struct CXTranslationUnitImpl {
  void *TUData;
  void *StringPool;
};
}

namespace clang {
  class ASTUnit;

namespace cxtu {

CXTranslationUnitImpl *MakeCXTranslationUnit(ASTUnit *TU);

}} // end namespace clang::cxtu

#endif
