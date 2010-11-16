//===- CXString.cpp - Routines for manipulating CXStrings -----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines routines for manipulating CXStrings. It should be the
// only file that has internal knowledge of the encoding of the data in
// CXStrings.
//
//===----------------------------------------------------------------------===//

#include "CXString.h"
#include "clang/Frontend/ASTUnit.h"
#include "clang-c/Index.h"
#include "llvm/Support/ErrorHandling.h"

using namespace clang;
using namespace clang::cxstring;

enum CXStringFlag { CXS_Unmanaged, CXS_Malloc };

CXString cxstring::createCXString(const char *String, bool DupString){
  CXString Str;
  if (DupString) {
    Str.Spelling = strdup(String);
    Str.private_flags = (unsigned) CXS_Malloc;
  } else {
    Str.Spelling = String;
    Str.private_flags = (unsigned) CXS_Unmanaged;
  }
  return Str;
}

CXString cxstring::createCXString(llvm::StringRef String, bool DupString) {
  CXString Result;
  if (DupString || (!String.empty() && String.data()[String.size()] != 0)) {
    char *Spelling = (char *)malloc(String.size() + 1);
    memmove(Spelling, String.data(), String.size());
    Spelling[String.size()] = 0;
    Result.Spelling = Spelling;
    Result.private_flags = (unsigned) CXS_Malloc;
  } else {
    Result.Spelling = String.data();
    Result.private_flags = (unsigned) CXS_Unmanaged;
  }
  return Result;
}

//===----------------------------------------------------------------------===//
// libClang public APIs.
//===----------------------------------------------------------------------===//

extern "C" {
const char *clang_getCString(CXString string) {
  return string.Spelling;
}

void clang_disposeString(CXString string) {
  if (string.private_flags == CXS_Malloc && string.Spelling)
    free((void*)string.Spelling);
}
} // end: extern "C"

