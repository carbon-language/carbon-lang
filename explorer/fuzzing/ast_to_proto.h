// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_EXPLORER_FUZZING_AST_TO_PROTO_H_
#define CARBON_EXPLORER_FUZZING_AST_TO_PROTO_H_

#include "explorer/ast/ast.h"
#include "testing/fuzzing/carbon.pb.h"

namespace Carbon::Testing {

// Builds a protobuf representation of `ast`.
auto AstToProto(const AST& ast) -> Fuzzing::Carbon;

}  // namespace Carbon::Testing

#endif  // CARBON_EXPLORER_FUZZING_AST_TO_PROTO_H_
