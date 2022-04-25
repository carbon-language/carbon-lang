// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXECUTABLE_SEMANTICS_FUZZING_FUZZER_UTIL_H_
#define EXECUTABLE_SEMANTICS_FUZZING_FUZZER_UTIL_H_

#include "common/fuzzing/carbon.pb.h"

namespace Carbon {

// Converts `compilation_unit` to Carbon. Adds an default `Main()`
// definition if one is not present in the proto.
auto ProtoToCarbonWithMain(const Fuzzing::CompilationUnit& compilation_unit)
    -> std::string;

// Parses and executes a fuzzer-generated program.
void ParseAndExecute(const Fuzzing::CompilationUnit& compilation_unit);

}  // namespace Carbon

#endif  // EXECUTABLE_SEMANTICS_FUZZING_FUZZER_UTIL_H_
