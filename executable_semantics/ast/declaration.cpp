// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "executable_semantics/ast/declaration.h"

namespace Carbon {

auto Declaration::MakeFunctionDeclaration(FunctionDefinition definition)
    -> const Declaration {
  Declaration d;
  d.value = FunctionDeclaration({.definition = definition});
  return d;
}

auto Declaration::MakeStructDeclaration(int line_num, std::string name,
                                        std::list<Member*> members)
    -> const Declaration {
  Declaration d;
  d.value = StructDeclaration(
      {.definition = StructDefinition({.line_num = line_num,
                                       .name = std::move(name),
                                       .members = std::move(members)})});
  return d;
}

auto Declaration::MakeChoiceDeclaration(
    int line_num, std::string name,
    std::list<std::pair<std::string, const Expression*>> alternatives)
    -> const Declaration {
  Declaration d;
  d.value = ChoiceDeclaration({.line_num = line_num,
                               .name = std::move(name),
                               .alternatives = std::move(alternatives)});
  return d;
}

auto Declaration::MakeVariableDeclaration(int source_location,
                                          const BindingPattern* binding,
                                          const Expression* initializer)
    -> const Declaration {
  Declaration d;
  d.value = VariableDeclaration({.source_location = source_location,
                                 .binding = binding,
                                 .initializer = initializer});
  return d;
}

auto Declaration::GetFunctionDeclaration() const -> const FunctionDeclaration& {
  return std::get<FunctionDeclaration>(value);
}

auto Declaration::GetStructDeclaration() const -> const StructDeclaration& {
  return std::get<StructDeclaration>(value);
}

auto Declaration::GetChoiceDeclaration() const -> const ChoiceDeclaration& {
  return std::get<ChoiceDeclaration>(value);
}

auto Declaration::GetVariableDeclaration() const -> const VariableDeclaration& {
  return std::get<VariableDeclaration>(value);
}

void Declaration::Print(llvm::raw_ostream& out) const {
  switch (tag()) {
    case DeclarationKind::FunctionDeclaration:
      out << GetFunctionDeclaration().definition;
      break;

    case DeclarationKind::StructDeclaration: {
      const StructDefinition& struct_def = GetStructDeclaration().definition;
      out << "struct " << struct_def.name << " {\n";
      for (Member* m : struct_def.members) {
        out << *m;
      }
      out << "}\n";
      break;
    }

    case DeclarationKind::ChoiceDeclaration: {
      const auto& choice = GetChoiceDeclaration();
      out << "choice " << choice.name << " {\n";
      for (const auto& [name, signature] : choice.alternatives) {
        out << "alt " << name << " " << *signature << ";\n";
      }
      out << "}\n";
      break;
    }

    case DeclarationKind::VariableDeclaration: {
      const auto& var = GetVariableDeclaration();
      out << "var " << *var.binding << " = " << *var.initializer << "\n";
      break;
    }
  }
}

}  // namespace Carbon
