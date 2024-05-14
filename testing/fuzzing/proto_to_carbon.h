// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TESTING_FUZZING_PROTO_TO_CARBON_H_
#define CARBON_TESTING_FUZZING_PROTO_TO_CARBON_H_

#include "common/error.h"
#include "testing/fuzzing/carbon.pb.h"

namespace Carbon {

// Builds a Carbon source from `compilation_unit`. The logic tries to produce a
// syntactially valid Carbon source for all cases, even if the input protocol
// buffer is invalid (like a variable declaration with an empty `name` field).
// This is done to reduce the number of inputs the fuzzer framework generates
// when trying to produce lexically valid source.
auto ProtoToCarbon(const Fuzzing::Carbon& proto, bool maybe_add_main)
    -> std::string;

// Parses the textproto into a proto object.
auto ParseCarbonTextProto(const std::string& contents)
    -> ErrorOr<Fuzzing::Carbon>;

}  // namespace Carbon

#endif  // CARBON_TESTING_FUZZING_PROTO_TO_CARBON_H_
