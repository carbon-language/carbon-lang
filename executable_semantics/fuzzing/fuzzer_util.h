// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef THIRD_PARTY_CARBON_LANG_EXECUTABLE_SEMANTICS_FUZZING_FUZZER_UTIL_H_
#define THIRD_PARTY_CARBON_LANG_EXECUTABLE_SEMANTICS_FUZZING_FUZZER_UTIL_H_

#include "common/fuzzing/carbon.pb.h"

namespace Carbon {

// Adds an empty `Main()` declaration if `compilation_unit` does not have one.
auto MaybeAddMain(Fuzzing::CompilationUnit& compilation_unit) -> void;

}  // namespace Carbon

#endif  // THIRD_PARTY_CARBON_LANG_EXECUTABLE_SEMANTICS_FUZZING_FUZZER_UTIL_H_
