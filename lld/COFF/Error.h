//===- Error.h --------------------------------------------------*- C++ -*-===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_COFF_ERROR_H
#define LLD_COFF_ERROR_H

#include "lld/Core/LLVM.h"
#include "llvm/Support/Error.h"

namespace lld {
namespace coff {

LLVM_ATTRIBUTE_NORETURN void fatal(const Twine &Msg);
void check(std::error_code EC, const Twine &Prefix);
void check(llvm::Error E, const Twine &Prefix);

template <typename T> void check(const ErrorOr<T> &V, const Twine &Prefix) {
  check(V.getError(), Prefix);
}

template <class T> T check(Expected<T> E, const Twine &Prefix) {
  if (E)
    return std::move(*E);
  fatal(E.takeError(), Prefix);
  return T();
}

} // namespace coff
} // namespace lld

#endif
