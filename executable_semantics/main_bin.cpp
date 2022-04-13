// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "executable_semantics/main.h"

auto main(int argc, char** argv) -> int {
  return Carbon::ExecutableSemanticsMain(
      "executable_semantics/data/prelude.carbon", argc, argv);
}
