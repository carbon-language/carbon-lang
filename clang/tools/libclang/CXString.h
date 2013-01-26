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
#include "clang/Basic/LLVM.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include <vector>

namespace clang {
namespace cxstring {

struct CXStringBuf;

/// \brief Create a CXString object from a C string.
CXString createCXString(const char *String, bool DupString = false);

/// \brief Create a CXString object from a StringRef.
CXString createCXString(StringRef String, bool DupString = true);

/// \brief Create a CXString object that is backed by a string buffer.
CXString createCXString(CXStringBuf *buf);

/// \brief A string pool used for fast allocation/deallocation of strings.
class CXStringPool {
public:
  ~CXStringPool();

  CXStringBuf *getCXStringBuf(CXTranslationUnit TU);

private:
  std::vector<CXStringBuf *> Pool;

  friend struct CXStringBuf;
};

struct CXStringBuf {
  SmallString<128> Data;
  CXTranslationUnit TU;

  CXStringBuf(CXTranslationUnit TU) : TU(TU) {}

  /// \brief Return this buffer to the pool.
  void dispose();
};

CXStringBuf *getCXStringBuf(CXTranslationUnit TU);

/// \brief Returns true if the CXString data is managed by a pool.
bool isManagedByPool(CXString str);

}
}

#endif

