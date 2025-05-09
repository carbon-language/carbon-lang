// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_COMMON_BUILD_DATA_H_
#define CARBON_COMMON_BUILD_DATA_H_

#include "common/build_data_linkstamp.h"
#include "llvm/ADT/StringRef.h"

namespace Carbon::BuildData {

// Build information for a binary, from bazel. Stamped values come from:
// https://github.com/bazelbuild/bazel/blob/master/src/main/java/com/google/devtools/build/lib/rules/cpp/CppLinkstampCompileHelper.java

// NOLINTBEGIN(readability-identifier-naming): We want to use constant-style
// names for the public variables, but cannot use constexpr.

// The platform, per https://bazel.build/extending/platforms.
inline const llvm::StringRef Platform = Internal::platform;

// Whether coverage is enabled.
inline const bool BuildCoverageEnabled = Internal::build_coverage_enabled;

// The binary target, such as `//common:build_data_test`.
inline const llvm::StringRef TargetName = Internal::target_name;

// The path to the build target, such as
// `bazel-out/k8-fastbuild/bin/common/build_data_test`.
inline const llvm::StringRef BuildTarget = Internal::build_target;

// NOLINTEND(readability-identifier-naming)

}  // namespace Carbon::BuildData

#endif  // CARBON_COMMON_BUILD_DATA_H_
