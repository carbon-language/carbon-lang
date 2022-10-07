// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_EXPLORER_MAIN_H_
#define CARBON_EXPLORER_MAIN_H_

#include "llvm/ADT/StringRef.h"

namespace Carbon {

// Runs explorer. relative_prelude_path must be POSIX-style, not native, and
// will be translated to native.
auto ExplorerMain(int argc, char** argv, void* static_for_main_addr,
                  llvm::StringRef relative_prelude_path) -> int;

}  // namespace Carbon

#endif  // CARBON_EXPLORER_MAIN_H_
