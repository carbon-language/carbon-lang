// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXECUTABLE_SEMANTICS_FUZZING_AST_TO_PROTO_H_
#define EXECUTABLE_SEMANTICS_FUZZING_AST_TO_PROTO_H_

#include "common/fuzzing/carbon.pb.h"
#include "executable_semantics/ast/ast.h"

namespace Carbon {

// Builds a protobuf representation of `ast`.
auto AstToProto(const AST& ast) -> Fuzzing::CompilationUnit;

}  // namespace Carbon

#endif  // EXECUTABLE_SEMANTICS_FUZZING_AST_TO_PROTO_H_
