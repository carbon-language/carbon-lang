// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef THIRD_PARTY_CARBON_LANG_EXECUTABLE_SEMANTICS_FUZZING_AST_TO_PROTO_H_
#define THIRD_PARTY_CARBON_LANG_EXECUTABLE_SEMANTICS_FUZZING_AST_TO_PROTO_H_

#include "executable_semantics/ast/ast.h"
#include "fuzzing/carbon.pb.h"

namespace Carbon {

// Builds a protobuf representation of `ast`.
auto ASTToProto(const AST& ast) -> Fuzzing::CompilationUnit;

}  // namespace Carbon

#endif  // THIRD_PARTY_CARBON_LANG_EXECUTABLE_SEMANTICS_FUZZING_AST_TO_PROTO_H_
