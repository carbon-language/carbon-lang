// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXECUTABLE_SEMANTICS_INTERPRETER_TYPECHECK_H_
#define EXECUTABLE_SEMANTICS_INTERPRETER_TYPECHECK_H_

#include <set>

#include "common/ostream.h"
#include "executable_semantics/ast/expression.h"
#include "executable_semantics/ast/statement.h"
#include "executable_semantics/interpreter/dictionary.h"
#include "executable_semantics/interpreter/interpreter.h"

namespace Carbon {

using TypeEnv = Dictionary<std::string, const Value*>;

struct TCExpression {
  TCExpression(const Expression* e, const Value* t, TypeEnv types)
      : exp(e), type(t), types(types) {}

  const Expression* exp;
  const Value* type;
  TypeEnv types;
};

struct TCPattern {
  const Pattern* pattern;
  const Value* type;
  TypeEnv types;
};

struct TCStatement {
  TCStatement(const Statement* s, TypeEnv types) : stmt(s), types(types) {}

  const Statement* stmt;
  TypeEnv types;
};

auto TypeCheckExp(const Expression* e, TypeEnv types, Env values)
    -> TCExpression;
auto TypeCheckPattern(const Pattern* p, TypeEnv types, Env values,
                      const Value* expected) -> TCPattern;

auto TypeCheckStmt(const Statement* s, TypeEnv types, Env values,
                   const Value*& ret_type, bool is_omitted_ret_type)
    -> TCStatement;

auto MakeTypeChecked(const Declaration& decl, const TypeEnv& types,
                     const Env& values) -> Declaration;
auto TopLevel(std::list<Declaration>* fs) -> TypeCheckContext;

}  // namespace Carbon

#endif  // EXECUTABLE_SEMANTICS_INTERPRETER_TYPECHECK_H_
