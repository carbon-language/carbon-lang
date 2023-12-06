// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/init_llvm.h"

#include "llvm/Support/TargetSelect.h"

namespace Carbon {

InitLLVM::InitLLVM(int& argc, char**& argv)
    : init_llvm(argc, argv), args(argv, argv + argc) {
  // LLVM assumes that argc and argv won't change, and registers them with an
  // `llvm::PrettyStackTraceProgram` that will crash if an argv element gets
  // nulled out, which for example `testing::InitGoogleTest` does. So make a
  // copy of argv for use by the program to satisfy LLVM's assumptions.
  argc = args.size();
  argv = args.data();

  llvm::setBugReportMsg(
      "Please report issues to "
      "https://github.com/carbon-language/carbon-lang/issues and include the "
      "crash backtrace.\n");
}

}  // namespace Carbon
