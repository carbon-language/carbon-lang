//===- llvm/Support/StringSaver.h - Stable storage for strings --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_STRINGSAVER_H
#define LLVM_SUPPORT_STRINGSAVER_H

#include "llvm/Support/Allocator.h"
#include <cstring>

namespace llvm {

/// \brief Saves strings in stable storage that it owns.
class StringSaver {
  BumpPtrAllocator Alloc;

public:
  const char *saveCStr(const char *CStr) {
    auto Len = std::strlen(CStr) + 1; // Don't forget the NUL!
    char *Buf = Alloc.Allocate<char>(Len);
    std::memcpy(Buf, CStr, Len);
    return Buf;
  }
};

} // end namespace llvm

#endif
