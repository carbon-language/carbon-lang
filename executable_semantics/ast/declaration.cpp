// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "executable_semantics/ast/declaration.h"

#include "llvm/Support/Casting.h"

namespace Carbon {

using llvm::cast;

void Declaration::Print(llvm::raw_ostream& out) const {
  switch (tag()) {
    case Kind::FunctionDeclaration:
      out << cast<FunctionDeclaration>(*this).definition();
      break;

    case Kind::ClassDeclaration: {
      const ClassDefinition& class_def =
          cast<ClassDeclaration>(*this).definition();
      out << "class " << class_def.name() << " {\n";
      for (Nonnull<Member*> m : class_def.members()) {
        out << *m;
      }
      out << "}\n";
      break;
    }

    case Kind::ChoiceDeclaration: {
      const auto& choice = cast<ChoiceDeclaration>(*this);
      out << "choice " << choice.name() << " {\n";
      for (const auto& alt : choice.alternatives()) {
        out << "alt " << alt.name() << " " << alt.signature() << ";\n";
      }
      out << "}\n";
      break;
    }

    case Kind::VariableDeclaration: {
      const auto& var = cast<VariableDeclaration>(*this);
      out << "var " << var.binding() << " = " << var.initializer() << "\n";
      break;
    }
  }
}

}  // namespace Carbon
