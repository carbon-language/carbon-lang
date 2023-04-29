// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_EXPLORER_FUZZING_FUZZER_UTIL_H_
#define CARBON_EXPLORER_FUZZING_FUZZER_UTIL_H_

#include "common/error.h"
#include "common/fuzzing/carbon.pb.h"
#include "explorer/ast/ast.h"

namespace Carbon {

// Parses and executes a fuzzer-generated program.
// Returns program result if execution was successful.
auto ParseAndExecute(const Fuzzing::Carbon& carbon) -> ErrorOr<int>;

// Returns a full path for a file under bazel runfiles.
// Exposed for testing.
auto GetRunfilesFile(const std::string& file) -> ErrorOr<std::string>;

}  // namespace Carbon

#endif  // CARBON_EXPLORER_FUZZING_FUZZER_UTIL_H_
