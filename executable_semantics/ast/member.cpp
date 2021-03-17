// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "executable_semantics/ast/member.h"

#include <iostream>

namespace Carbon {

auto MakeField(int line_num, std::string name, Expression type) -> Member* {
  return new Member{line_num,
                    MemberKind::FieldMember,
                    {new std::string(std::move(name)), new Expression(type)}};
}

void PrintMember(Member* m) {
  switch (m->tag) {
    case MemberKind::FieldMember:
      std::cout << "var " << *m->u.field.name << " : ";
      m->u.field.type->Print();
      std::cout << ";" << std::endl;
      break;
  }
}

}  // namespace Carbon
