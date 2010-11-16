//===- CXString.h - Routines for manipulating CXStrings -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines routines for manipulating CXStrings.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_CXSTRING_H
#define LLVM_CLANG_CXSTRING_H

#include "clang-c/Index.h"
#include "llvm/ADT/StringRef.h"

namespace clang {
namespace cxstring {

/// \brief Create a CXString object from a C string.
CXString createCXString(const char *String, bool DupString = false);

/// \brief Create a CXString ojbect from a StringRef.
CXString createCXString(llvm::StringRef String, bool DupString = true);  

}
}

#endif

