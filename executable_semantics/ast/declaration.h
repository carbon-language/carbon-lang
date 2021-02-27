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

struct Value;
template <class K, class V>
struct AssocList;
using Address = unsigned int;
using TypeEnv = AssocList<std::string, Value*>;
using Env = AssocList<std::string, Address>;

/// TODO:explain this. Also name it if necessary. Consult with jsiek.
using ExecutionEnvironment = std::pair<TypeEnv*, Env*>;

struct Declaration {
  virtual void Print() const = 0;
  virtual auto Name() const -> std::string = 0;
  virtual auto TypeChecked(TypeEnv* env, Env* ct_env) const
      -> const Declaration* = 0;
  virtual void InitGlobals(Env*& globals) const = 0;
  virtual auto TopLevel(ExecutionEnvironment&) const -> void = 0;
};

struct FunctionDeclaration : Declaration {
  const FunctionDefinition* definition;
  explicit FunctionDeclaration(const FunctionDefinition* definition)
      : definition(definition) {}

  void Print() const;
  auto Name() const -> std::string;
  auto TypeChecked(TypeEnv* env, Env* ct_env) const -> const Declaration*;
  void InitGlobals(Env*& globals) const;
  auto TopLevel(ExecutionEnvironment&) const -> void;
};

struct StructDeclaration : Declaration {
  StructDefinition definition;
  StructDeclaration(int line_num, std::string name, std::list<Member*>* members)
      : definition{line_num, new std::string(name), members} {}

  void Print() const;
  auto Name() const -> std::string;
  auto TypeChecked(TypeEnv* env, Env* ct_env) const -> const Declaration*;
  void InitGlobals(Env*& globals) const;
  auto TopLevel(ExecutionEnvironment&) const -> void;
};

struct ChoiceDeclaration : Declaration {
  int line_num;
 std::string name;
  std::list<std::pair<std::string, Expression*>> alternatives;

  ChoiceDeclaration(int line_num, std::string name,
                    std::list<std::pair<std::string, Expression*>> alternatives)
      : line_num(line_num), name(name), alternatives(alternatives) {}

  void Print() const;
  auto Name() const -> std::string;
  auto TypeChecked(TypeEnv* env, Env* ct_env) const -> const Declaration*;
  void InitGlobals(Env*& globals) const;
  auto TopLevel(ExecutionEnvironment&) const -> void;
};

}  // namespace Carbon

#endif  // EXECUTABLE_SEMANTICS_AST_DECLARATION_H_
