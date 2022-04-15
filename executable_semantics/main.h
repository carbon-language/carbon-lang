// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXECUTABLE_SEMANTICS_MAIN_H_
#define EXECUTABLE_SEMANTICS_MAIN_H_

#include "llvm/ADT/StringRef.h"

namespace Carbon {

// Runs executable semantics.
auto ExecutableSemanticsMain(llvm::StringRef default_prelude_file, int argc,
                             char** argv) -> int;

}  // namespace Carbon

#endif  // EXECUTABLE_SEMANTICS_MAIN_H_
