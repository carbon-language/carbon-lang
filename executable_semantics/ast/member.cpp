// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "executable_semantics/ast/member.h"

#include <iostream>

namespace Carbon {

auto MakeField(int line_num, std::string name, Expression type) -> Member* {
  return new Member{line_num, name, type};
}

void PrintMember(Member* m) {
  std::cout << "var " << m->name << " : ";
  m->type.Print();
  std::cout << ";" << std::endl;
}

}  // namespace Carbon
