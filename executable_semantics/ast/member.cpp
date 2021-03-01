// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "executable_semantics/ast/member.h"

#include <iostream>

namespace Carbon {

auto MakeField(int line_num, std::string name, Expression* type) -> Member* {
  auto m = new Member();
  m->line_num = line_num;
  m->tag = MemberKind::FieldMember;
  m->u.field.name = new std::string(std::move(name));
  m->u.field.type = type;
  return m;
}

void PrintMember(Member* m) {
  switch (m->tag) {
    case MemberKind::FieldMember:
      std::cout << "var " << *m->u.field.name << " : " << *m->u.field.type
                << ";" << std::endl;
      break;
  }
}

}  // namespace Carbon
