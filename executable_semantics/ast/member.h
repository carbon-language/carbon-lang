// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXECUTABLE_SEMANTICS_AST_MEMBER_H_
#define EXECUTABLE_SEMANTICS_AST_MEMBER_H_

#include <string>

#include "common/ostream.h"
#include "executable_semantics/ast/expression.h"
#include "executable_semantics/ast/pattern.h"
#include "llvm/Support/Compiler.h"

namespace Carbon {

enum class MemberKind { FieldMember };

struct FieldMember {
  static constexpr MemberKind Kind = MemberKind::FieldMember;
  // TODO: split this into a non-optional name and a type, initialized by
  // a constructor that takes a BindingPattern and handles errors like a
  // missing name.
  const BindingPattern* binding;
};

struct Member {
  static auto MakeFieldMember(int line_num, const BindingPattern* binding)
      -> Member*;

  auto GetFieldMember() const -> const FieldMember&;

  void Print(llvm::raw_ostream& out) const;
  LLVM_DUMP_METHOD void Dump() const { Print(llvm::errs()); }

  inline auto tag() const -> MemberKind {
    return std::visit([](const auto& t) { return t.Kind; }, value);
  }

  int line_num;

 private:
  std::variant<FieldMember> value;
};

}  // namespace Carbon

#endif  // EXECUTABLE_SEMANTICS_AST_MEMBER_H_
