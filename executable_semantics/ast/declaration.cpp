// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "executable_semantics/ast/declaration.h"

#include <iostream>

namespace Carbon {

void FunctionDeclaration::Print() const { definition.Print(); }

void StructDeclaration::Print() const {
  std::cout << "struct " << *definition.name << " {" << std::endl;
  for (auto& member : *definition.members) {
    member->Print();
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
