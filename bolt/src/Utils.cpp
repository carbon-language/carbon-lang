//===--- Utils.cpp - Common helper functions ------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Common helper functions.
//
//===----------------------------------------------------------------------===//

#include "Utils.h"
#include "llvm/Support/raw_ostream.h"

namespace llvm {
namespace bolt {

void report_error(StringRef Message, std::error_code EC) {
  assert(EC);
  errs() << "BOLT-ERROR: '" << Message << "': " << EC.message() << ".\n";
  exit(1);
}

void report_error(StringRef Message, Error E) {
  assert(E);
  errs() << "BOLT-ERROR: '" << Message << "': " << toString(std::move(E))
         << ".\n";
  exit(1);
}

void check_error(std::error_code EC, StringRef Message) {
  if (!EC)
    return;
  report_error(Message, EC);
}

void check_error(Error E, Twine Message) {
  if (!E)
    return;
  handleAllErrors(std::move(E), [&](const llvm::ErrorInfoBase &EIB) {
    llvm::errs() << "BOLT-ERROR: '" << Message << "': " << EIB.message()
                 << '\n';
    exit(1);
  });
}

} // namespace bolt
} // namespace llvm
