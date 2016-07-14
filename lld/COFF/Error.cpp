//===- Error.cpp ----------------------------------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Error.h"

#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_ostream.h"

namespace lld {
namespace coff {

void fatal(const Twine &Msg) {
  llvm::errs() << Msg << "\n";
  exit(1);
}

void check(std::error_code EC, const Twine &Prefix) {
  if (!EC)
    return;
  fatal(Prefix + ": " + EC.message());
}

void check(llvm::Error E, const Twine &Prefix) {
  if (!E)
    return;
  handleAllErrors(std::move(E), [&](const llvm::ErrorInfoBase &EIB) {
    fatal(Prefix + ": " + EIB.message());
  });
}

} // namespace coff
} // namespace lld
