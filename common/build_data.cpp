// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/build_data.h"

namespace Carbon::BuildData {

const llvm::StringRef Platform = Internal::platform;

// Whether coverage is enabled.
const bool BuildCoverageEnabled = Internal::build_coverage_enabled;

// The binary target, such as `//common:build_data_test`.
const llvm::StringRef TargetName = Internal::target_name;

// The path to the build target, such as
// `bazel-out/k8-fastbuild/bin/common/build_data_test`.
const llvm::StringRef BuildTarget = Internal::build_target;

}  // namespace Carbon::BuildData
