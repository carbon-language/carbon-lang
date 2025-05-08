// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_COMMON_BUILD_DATA_H_
#define CARBON_COMMON_BUILD_DATA_H_

#include "llvm/ADT/StringRef.h"

namespace Carbon {

// Build information for a binary, from bazel. Stamped values come from:
// https://github.com/bazelbuild/bazel/blob/master/src/main/java/com/google/devtools/build/lib/rules/cpp/CppLinkstampCompileHelper.java
struct BuildData {
  // The platform, per https://bazel.build/extending/platforms.
  static const llvm::StringRef Platform;

  // Whether coverage is enabled.
  static const bool BuildCoverageEnabled;

  // The binary target, such as `//common:build_data_test`.
  static const llvm::StringRef TargetName;

  // The path to the build target, such as
  // `bazel-out/k8-fastbuild/bin/common/build_data_test`.
  static const llvm::StringRef BuildTarget;
};

}  // namespace Carbon

#endif  // CARBON_COMMON_BUILD_DATA_H_
