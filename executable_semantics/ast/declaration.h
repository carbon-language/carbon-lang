// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXECUTABLE_SEMANTICS_AST_DECLARATION_H_
#define EXECUTABLE_SEMANTICS_AST_DECLARATION_H_

#include <list>
#include <string>

#include "executable_semantics/ast/function_definition.h"
#include "executable_semantics/ast/member.h"
#include "executable_semantics/ast/struct_definition.h"

namespace Carbon {

enum class DeclarationKind {
  FunctionDeclaration,
  StructDeclaration,
  ChoiceDeclaration
};

struct Declaration {
  DeclarationKind tag;
  union {
    struct FunctionDefinition* fun_def;
    struct StructDefinition* struct_def;
    struct {
      int line_num;
      std::string* name;
      std::list<std::pair<std::string, Expression*>>* alternatives;
    } choice_def;
  } u;
};

auto MakeFunDecl(struct FunctionDefinition* f) -> Declaration*;
auto MakeStructDecl(int line_num, std::string name, std::list<Member*>* members)
    -> Declaration*;
auto MakeChoiceDecl(int line_num, std::string name,
                    std::list<std::pair<std::string, Expression*>>* alts)
    -> Declaration*;

void PrintDecl(Declaration* d);

}  // namespace Carbon

#endif  // EXECUTABLE_SEMANTICS_AST_DECLARATION_H_
