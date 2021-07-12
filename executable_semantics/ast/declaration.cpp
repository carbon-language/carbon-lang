// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "executable_semantics/ast/declaration.h"

#include <iostream>

namespace Carbon {

struct TagVisitor {
  template <typename Alternative>
  auto operator()(const Alternative&) -> DeclarationKind {
    return Alternative::Kind;
  }
};

}  // namespace Carbon

auto Declaration::tag() const -> DeclarationKind {
  return std::visit(TagVisitor(), value);
}

static auto MakeFunctionDeclaration() -> const Declaration* {
  Declaration* d = new Declaration();
  d->value = FunctionDeclaration();
  return d;
}

static auto MakeStructDeclaration() -> const Declaration* {
  Declaration* d = new Declaration();
  d->value = StructDeclaration();
  return d;
}

static auto MakeChoiceDeclaration() -> const Declaration* {
  Declaration* d = new Declaration();
  d->value = ChoiceDeclaration();
  return d;
}

static auto MakeVariableDeclaration() -> const Declaration* {
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

void FunctionDeclaration::Print() const { PrintFunDef(definition); }

void StructDeclaration::Print() const {
  std::cout << "struct " << *definition.name << " {" << std::endl;
  for (auto& member : *definition.members) {
    PrintMember(member);
  }
  std::cout << "}" << std::endl;
}

void ChoiceDeclaration::Print() const {
  std::cout << "choice " << name << " {" << std::endl;
  for (const auto& [name, signature] : alternatives) {
    std::cout << "alt " << name << " ";
    PrintExp(signature);
    std::cout << ";" << std::endl;
  }
  std::cout << "}" << std::endl;
}

// Print a global variable declaration to standard out.
void VariableDeclaration::Print() const {
  std::cout << "var ";
  PrintExp(type);
  std::cout << " : " << name << " = ";
  PrintExp(initializer);
  std::cout << std::endl;
}

}  // namespace Carbon
