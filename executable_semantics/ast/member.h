// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXECUTABLE_SEMANTICS_AST_MEMBER_H_
#define EXECUTABLE_SEMANTICS_AST_MEMBER_H_

#include <string>

#include "executable_semantics/ast/expression.h"

namespace Carbon {

enum class MemberKind { FieldMember };

struct Member {
  int line_num;
  MemberKind tag;
  union {
    struct {
      std::string* name;
      Expression* type;
    } field;
  } u;
};

auto MakeField(int line_num, std::string name, Expression* type) -> Member*;

void PrintMember(Member* m);

}  // namespace Carbon

#endif  // EXECUTABLE_SEMANTICS_AST_MEMBER_H_
