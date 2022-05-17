// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_EXPLORER_MAIN_H_
#define CARBON_EXPLORER_MAIN_H_

#include "llvm/ADT/StringRef.h"

namespace Carbon {

// Runs explorer.
auto ExplorerMain(llvm::StringRef default_prelude_file, int argc, char** argv)
    -> int;

}  // namespace Carbon

#endif  // CARBON_EXPLORER_MAIN_H_
