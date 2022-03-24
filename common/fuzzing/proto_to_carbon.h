// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef THIRD_PARTY_CARBON_LANG_EXECUTABLE_SEMANTICS_FUZZING_PROTO_TO_CARBON_H_
#define THIRD_PARTY_CARBON_LANG_EXECUTABLE_SEMANTICS_FUZZING_PROTO_TO_CARBON_H_

#include "common/fuzzing/carbon.pb.h"

namespace Carbon {

// Builds a Carbon source from `compilation_unit`.
auto ProtoToCarbon(const Fuzzing::CompilationUnit& compilation_unit)
    -> std::string;

}  // namespace Carbon

#endif  // THIRD_PARTY_CARBON_LANG_EXECUTABLE_SEMANTICS_FUZZING_PROTO_TO_CARBON_H_
