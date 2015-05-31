//===- Memory.h -----------------------------------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_COFF_MEMORY_H
#define LLD_COFF_MEMORY_H

#include "lld/Core/LLVM.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Allocator.h"
#include <memory>

namespace lld {
namespace coff {

class StringAllocator {
public:
  // Returns a null-terminated copy of a string.
  StringRef save(StringRef S) {
    char *P = Alloc.Allocate<char>(S.size() + 1);
    memcpy(P, S.data(), S.size());
    P[S.size()] = '\0';
    return StringRef(P, S.size());
  }

  StringRef save(Twine S) { return save(StringRef(S.str())); }
  StringRef save(const char *S) { return save(StringRef(S)); }
  StringRef save(std::string &S) { return save(StringRef(S)); }

private:
  llvm::BumpPtrAllocator Alloc;
};

} // namespace coff
} // namespace lld

#endif
