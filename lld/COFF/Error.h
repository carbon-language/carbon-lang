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

namespace lld {
namespace coff {

LLVM_ATTRIBUTE_NORETURN void error(const Twine &Msg);
void error(std::error_code EC, const Twine &Prefix);

template <typename T> void error(const ErrorOr<T> &V, const Twine &Prefix) {
  error(V.getError(), Prefix);
}

} // namespace coff
} // namespace lld

#endif
