// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXECUTABLE_SEMANTICS_FUZZING_FUZZER_UTIL_H_
#define EXECUTABLE_SEMANTICS_FUZZING_FUZZER_UTIL_H_

#include "common/error.h"
#include "common/fuzzing/carbon.pb.h"

namespace Carbon {

// Converts `compilation_unit` to Carbon. Adds an default `Main()`
// definition if one is not present in the proto.
auto ProtoToCarbonWithMain(const Fuzzing::CompilationUnit& compilation_unit)
    -> std::string;

// Parses and executes a fuzzer-generated program.
void ParseAndExecute(const Fuzzing::CompilationUnit& compilation_unit);

namespace Internal {

// Returns a full path for a file under bazel runfiles.
// Exposed for testing.
auto GetRunfilesFile(const std::string& file) -> ErrorOr<std::string>;

}  // namespace Internal
}  // namespace Carbon

#endif  // EXECUTABLE_SEMANTICS_FUZZING_FUZZER_UTIL_H_
