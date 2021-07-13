// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "executable_semantics/ast/declaration.h"

#include <iostream>

namespace Carbon {

auto Declaration::MakeFunctionDeclaration(FunctionDefinition definition)
    -> const Declaration* {
  Declaration* d = new Declaration();
  d->value = FunctionDeclaration({.definition = definition});
  return d;
}

auto Declaration::MakeStructDeclaration() -> const Declaration* {
  Declaration* d = new Declaration();
  d->value = StructDeclaration();
  return d;
}

auto Declaration::MakeChoiceDeclaration() -> const Declaration* {
  Declaration* d = new Declaration();
  d->value = ChoiceDeclaration();
  return d;
}

auto Declaration::MakeVariableDeclaration() -> const Declaration* {
  Declaration* d = new Declaration();
  d->value = VariableDeclaration();
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

void Declaration::Print() {
  switch (tag()) {
    case DeclarationKind::FunctionDeclaration:
      PrintFunDef(GetFunctionDeclaration().definition);
      break;
    case DeclarationKind::StructDeclaration: {
      const auto& d = GetStructDeclaration();
      std::cout << "struct " << *d.definition.name << " {" << std::endl;
      for (auto& member : *d.definition.members) {
        PrintMember(member);
      }
      std::cout << "}" << std::endl;
      break;
    }
    case DeclarationKind::ChoiceDeclaration: {
      const auto& d = GetChoiceDeclaration();
      std::cout << "choice " << d.name << " {" << std::endl;
      for (const auto& [name, signature] : d.alternatives) {
        std::cout << "alt " << name << " ";
        PrintExp(signature);
        std::cout << ";" << std::endl;
      }
      std::cout << "}" << std::endl;
      break;
    }
    case DeclarationKind::VariableDeclaration: {
      const auto& d = GetVariableDeclaration();
      std::cout << "var ";
      PrintExp(d.type);
      std::cout << " : " << d.name << " = ";
      PrintExp(d.initializer);
      std::cout << std::endl;
      break;
    }
  }
}

}  // namespace Carbon
