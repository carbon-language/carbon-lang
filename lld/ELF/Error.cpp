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
namespace elf2 {

void warning(const Twine &Msg) { llvm::errs() << Msg << "\n"; }

void error(const Twine &Msg) {
  llvm::errs() << Msg << "\n";
  exit(1);
}

void error(std::error_code EC, const Twine &Prefix) {
  if (!EC)
    return;
  error(Prefix + ": " + EC.message());
}

void error(std::error_code EC) {
  if (!EC)
    return;
  error(EC.message());
}

} // namespace elf2
} // namespace lld
