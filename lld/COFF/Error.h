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
LLVM_ATTRIBUTE_NORETURN void fatal(std::error_code EC, const Twine &Prefix);
LLVM_ATTRIBUTE_NORETURN void fatal(llvm::Error &Err, const Twine &Prefix);

template <typename T> void check(const ErrorOr<T> &V, const Twine &Prefix) {
  if (auto EC = V.getError())
    fatal(EC, Prefix);
}

template <class T> T check(Expected<T> E, const Twine &Prefix) {
  if (E)
    return std::move(*E);
  fatal(E.takeError(), Prefix);
}

} // namespace coff
} // namespace lld

#endif
