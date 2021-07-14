// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "executable_semantics/ast/declaration.h"

#include <iostream>

namespace Carbon {

auto Declaration::MakeFunctionDeclaration(FunctionDefinition definition)
    -> const Declaration {
  Declaration d;
  d.value = FunctionDeclaration({.definition = definition});
  return d;
}

auto Declaration::MakeStructDeclaration(int line_num, std::string name,
                                        std::list<Member*>* members)
    -> const Declaration {
  Declaration d;
  d.value = StructDeclaration(
      {.definition = StructDefinition({.line_num = line_num,
                                       .name = new std::string(name),
                                       .members = members})});
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

auto Declaration::MakeVariableDeclaration(int source_location, std::string name,
                                          const Expression* type,
                                          const Expression* initializer)
    -> const Declaration {
  Declaration d;
  d.value = VariableDeclaration({.source_location = source_location,
                                 .name = std::move(name),
                                 .type = type,
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

void Declaration::Print() const {
  switch (tag()) {
    case DeclarationKind::FunctionDeclaration:
      PrintFunDef(GetFunctionDeclaration().definition);
      break;

    case DeclarationKind::StructDeclaration: {
      const auto& alt = GetStructDeclaration();
      std::cout << "struct " << *alt.definition.name << " {" << std::endl;
      for (auto& member : *alt.definition.members) {
        PrintMember(member);
      }
      std::cout << "}" << std::endl;
      break;
    }

    case DeclarationKind::ChoiceDeclaration: {
      const auto& alt = GetChoiceDeclaration();
      std::cout << "choice " << alt.name << " {" << std::endl;
      for (const auto& [name, signature] : alt.alternatives) {
        std::cout << "alt " << name << " ";
        PrintExp(signature);
        std::cout << ";" << std::endl;
      }
      std::cout << "}" << std::endl;
      break;
    }

    case DeclarationKind::VariableDeclaration: {
      const auto& alt = GetVariableDeclaration();
      std::cout << "var ";
      PrintExp(alt.type);
      std::cout << " : " << alt.name << " = ";
      PrintExp(alt.initializer);
      std::cout << std::endl;
      break;
    }
  }
}

}  // namespace Carbon
