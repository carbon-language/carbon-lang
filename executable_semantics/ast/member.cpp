// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "executable_semantics/ast/member.h"

#include "executable_semantics/common/arena.h"
#include "llvm/Support/Casting.h"

namespace Carbon {

using llvm::cast;

Member::~Member() = default;

void Member::Print(llvm::raw_ostream& out) const {
  switch (kind()) {
    case MemberKind::FieldMember: {
      const auto& field = cast<FieldMember>(*this);
      out << "var " << field.binding() << ";\n";
      break;
    }
    case MemberKind::ClassFunctionMember: {
      const auto& function = cast<ClassFunctionMember>(*this);
      out << "fn " << function.name() << " ";
      out << function.param_pattern() << function.return_term();
      if (function.body()) {
        out << " {\n";
        (*function.body())->Print(out);
        out << "\n}\n";
      } else {
        out << ";\n";
      }
      break;
    }
    case MemberKind::MethodMember: {
      const auto& method = cast<MethodMember>(*this);
      out << "fn " << method.name() << " ";
      out << method.me_pattern() << ".";
      out << method.param_pattern() << method.return_term();
      if (method.body()) {
        out << " {\n";
        (*method.body())->Print(out);
        out << "\n}\n";
      } else {
        out << ";\n";
      }
      break;
    }
  }
}

}  // namespace Carbon
