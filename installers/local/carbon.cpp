// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "executable_semantics/main.h"
#include "llvm/Support/Path.h"

auto main(int argc, char** argv) -> int {
  llvm::StringRef bin = llvm::sys::path::filename(argv[0]);
  if (bin == "carbon-cam") {
    return Carbon::ExecutableSemanticsMain(
        "/usr/lib/carbon/data/prelude.carbon", argc, argv);
  } else {
    fprintf(stderr, "Unrecognized Carbon binary requested: %s", argv[0]);
    return 1;
  }
}
