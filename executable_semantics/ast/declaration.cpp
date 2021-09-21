// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "executable_semantics/ast/declaration.h"

#include "llvm/Support/Casting.h"

namespace Carbon {

using llvm::cast;

void Declaration::Print(llvm::raw_ostream& out) const {
  switch (Tag()) {
    case Kind::FunctionDeclaration:
      out << cast<FunctionDeclaration>(*this).Definition();
      break;

    case Kind::ClassDeclaration: {
      const ClassDefinition& class_def =
          cast<ClassDeclaration>(*this).Definition();
      out << "class " << class_def.name << " {\n";
      for (Nonnull<Member*> m : class_def.members) {
        out << *m;
      }
      out << "}\n";
      break;
    }

    case Kind::ChoiceDeclaration: {
      const auto& choice = cast<ChoiceDeclaration>(*this);
      out << "choice " << choice.Name() << " {\n";
      for (const auto& [name, signature] : choice.Alternatives()) {
        out << "alt " << name << " " << *signature << ";\n";
      }
      out << "}\n";
      break;
    }

    case Kind::VariableDeclaration: {
      const auto& var = cast<VariableDeclaration>(*this);
      out << "var " << *var.Binding() << " = " << *var.Initializer() << "\n";
      break;
    }
  }
}

}  // namespace Carbon
