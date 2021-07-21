// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXECUTABLE_SEMANTICS_COMMON_ERROR_H_
#define EXECUTABLE_SEMANTICS_COMMON_ERROR_H_

#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/raw_ostream.h"

namespace Carbon {

// Prints an error and exits. This should be used for non-recoverable errors
// with user input.
//
// For example:
//   FatalUserError() << "Input is not valid!";
class FatalUserError {
 public:
  FatalUserError() { llvm::errs() << "ERROR: "; }
  ~FatalUserError() {
    // Finish with a newline.
    llvm::errs() << "\n";
    exit(-1);
  }

  // Forward output to llvm::errs.
  template <typename T>
  FatalUserError& operator<<(const T& message) {
    llvm::errs() << message;
    return *this;
  }
};

}  // namespace Carbon

#endif  // EXECUTABLE_SEMANTICS_COMMON_ERROR_H_
