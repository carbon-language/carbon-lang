// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "executable_semantics/ast/declaration.h"

#include <iostream>

namespace Carbon {

auto MakeFunDecl(FunctionDefinition* f) -> Declaration* {
  auto* d = new Declaration();
  d->tag = DeclarationKind::FunctionDeclaration;
  d->u.fun_def = f;
  return d;
}

auto MakeStructDecl(int line_num, std::string name, std::list<Member*>* members)
    -> Declaration* {
  auto* d = new Declaration();
  d->tag = DeclarationKind::StructDeclaration;
  d->u.struct_def = new StructDefinition();
  d->u.struct_def->line_num = line_num;
  d->u.struct_def->name = new std::string(std::move(name));
  d->u.struct_def->members = members;
  return d;
}

auto MakeChoiceDecl(int line_num, std::string name,
                    std::list<std::pair<std::string, Expression*>>* alts)
    -> Declaration* {
  auto* d = new Declaration();
  d->tag = DeclarationKind::ChoiceDeclaration;
  d->u.choice_def.line_num = line_num;
  d->u.choice_def.name = new std::string(std::move(name));
  d->u.choice_def.alternatives = alts;
  return d;
}

void PrintDecl(Declaration* d) {
  switch (d->tag) {
    case DeclarationKind::FunctionDeclaration:
      PrintFunDef(d->u.fun_def);
      break;
    case DeclarationKind::StructDeclaration:
      std::cout << "struct " << *d->u.struct_def->name << " {" << std::endl;
      for (auto& member : *d->u.struct_def->members) {
        PrintMember(member);
      }
      std::cout << "}" << std::endl;
      break;
    case DeclarationKind::ChoiceDeclaration:
      std::cout << "choice " << *d->u.choice_def.name << " {" << std::endl;
      for (auto& alternative : *d->u.choice_def.alternatives) {
        std::cout << "alt " << alternative.first << " ";
        PrintExp(alternative.second);
        std::cout << ";" << std::endl;
      }
      std::cout << "}" << std::endl;
      break;
  }
}

}  // namespace Carbon
