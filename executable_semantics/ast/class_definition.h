// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXECUTABLE_SEMANTICS_AST_CLASS_DEFINITION_H_
#define EXECUTABLE_SEMANTICS_AST_CLASS_DEFINITION_H_

#include <string>
#include <vector>

#include "executable_semantics/ast/member.h"
#include "executable_semantics/ast/source_location.h"

namespace Carbon {

struct ClassDefinition {
  SourceLocation loc;
  std::string name;
  std::vector<Ptr<Member>> members;
};

}  // namespace Carbon

#endif  // EXECUTABLE_SEMANTICS_AST_CLASS_DEFINITION_H_
