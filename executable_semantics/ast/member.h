// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXECUTABLE_SEMANTICS_AST_MEMBER_H_
#define EXECUTABLE_SEMANTICS_AST_MEMBER_H_

#include <string>

#include "common/ostream.h"
#include "executable_semantics/ast/expression.h"

namespace Carbon {

enum class MemberKind { FieldMember };

struct FieldMember {
  static constexpr MemberKind Kind = MemberKind::FieldMember;
  std::string name;
  const Expression* type;
};

struct Member {
  static auto MakeFieldMember(int line_num, std::string name,
                              const Expression* type) -> Member*;

  auto GetFieldMember() const -> const FieldMember&;

  void Print(llvm::raw_ostream& out) const;

  inline auto tag() const -> MemberKind {
    return std::visit([](const auto& t) { return t.Kind; }, value);
  }

  int line_num;

 private:
  std::variant<FieldMember> value;
};

}  // namespace Carbon

#endif  // EXECUTABLE_SEMANTICS_AST_MEMBER_H_
