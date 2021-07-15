// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXECUTABLE_SEMANTICS_AST_STRUCT_DEFINITION_H_
#define EXECUTABLE_SEMANTICS_AST_STRUCT_DEFINITION_H_

#include <list>
#include <string>

#include "executable_semantics/ast/member.h"

namespace Carbon {

struct StructDefinition {
  int line_num;
  std::string name;
  std::list<Member*> members;
};

}  // namespace Carbon

#endif  // EXECUTABLE_SEMANTICS_AST_STRUCT_DEFINITION_H_
