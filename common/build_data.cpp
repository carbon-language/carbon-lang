// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/build_data.h"

namespace Carbon::BuildData {

const llvm::StringRef Platform = Internal::platform;
const bool BuildCoverageEnabled = Internal::build_coverage_enabled;
const llvm::StringRef TargetName = Internal::target_name;
const llvm::StringRef BuildTarget = Internal::build_target;

}  // namespace Carbon::BuildData
