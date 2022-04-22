// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef COMMON_RUNFILES_UTIL_H_
#define COMMON_RUNFILES_UTIL_H_

#include <string>

namespace Carbon {

// Returns bazel's runfiles directory.
auto GetRunfilesDir() -> std::string;

}  // namespace Carbon
#endif  // COMMON_RUNFILES_UTIL_H_
