// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "executable_semantics/ast/member.h"

#include "executable_semantics/common/arena.h"

namespace Carbon {

auto Member::MakeFieldMember(int line_num, std::string name,
                             const Expression* type) -> Member* {
  auto m = global_arena->New<Member>();
  m->line_num = line_num;
  m->value = FieldMember({.name = std::move(name), .type = type});
  return m;
}

auto Member::GetFieldMember() const -> const FieldMember& {
  return std::get<FieldMember>(value);
}

void Member::Print(llvm::raw_ostream& out) const {
  switch (tag()) {
    case MemberKind::FieldMember:
      const auto& field = GetFieldMember();
      out << "var " << field.name << " : " << *field.type << ";\n";
      break;
  }
}

}  // namespace Carbon
