// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "testing/base/global_exe_path.h"

#include <string>

#include "common/check.h"
#include "common/exe_path.h"

static constinit std::optional<std::string> exe_path = {};

namespace Carbon::Testing {

auto GetExePath() -> llvm::StringRef {
  CARBON_CHECK(
      exe_path,
      "Must not query the executable path until after it has been set!");
  return *exe_path;
}

auto SetExePath(const char* argv_zero) -> void {
  CARBON_CHECK(!exe_path, "Must not call `SetExePath` more than once!");
  exe_path.emplace(Carbon::FindExecutablePath(argv_zero));
}

}  // namespace Carbon::Testing
