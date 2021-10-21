// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "executable_semantics/ast/member.h"

#include "executable_semantics/common/arena.h"
#include "llvm/Support/Casting.h"

namespace Carbon {

using llvm::cast;

void Member::Print(llvm::raw_ostream& out) const {
  switch (kind()) {
    case Kind::FieldMember:
      const auto& field = cast<FieldMember>(*this);
      out << "var " << field.binding() << ";\n";
      break;
  }
}

}  // namespace Carbon
