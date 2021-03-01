// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "executable_semantics/ast/declaration.h"

#include <iostream>

namespace Carbon {

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
  for (auto& alternative : alternatives) {
    std::cout << "alt " << alternative.first << " " << *alternative.second
              << ";" << std::endl;
  }
  std::cout << "}" << std::endl;
}

}  // namespace Carbon
