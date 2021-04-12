// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXECUTABLE_SEMANTICS_INTERPRETER_TYPECHECK_H_
#define EXECUTABLE_SEMANTICS_INTERPRETER_TYPECHECK_H_

#include <set>

#include "executable_semantics/ast/expression.h"
#include "executable_semantics/ast/statement.h"
#include "executable_semantics/interpreter/dictionary.h"
#include "executable_semantics/interpreter/interpreter.h"

namespace Carbon {

using TypeEnv = Dictionary<std::string, const Value*>;

void PrintTypeEnv(TypeEnv types);

enum class TCContext { ValueContext, PatternContext, TypeContext };

struct TCResult {
  TCResult(Expression* e, const Value* t, TypeEnv types)
      : exp(e), type(t), types(types) {}

  Expression* exp;
  const Value* type;
  TypeEnv types;
};

struct TCStatement {
  TCStatement(Statement* s, TypeEnv types) : stmt(s), types(types) {}

  Statement* stmt;
  TypeEnv types;
};

auto ToType(int line_num, const Value* val) -> const Value*;

auto TypeCheckExp(Expression* e, TypeEnv types, Env values,
                  const Value* expected, TCContext context) -> TCResult;

auto TypeCheckStmt(Statement*, TypeEnv, Env, Value const*&) -> TCStatement;

auto TypeCheckFunDef(struct FunctionDefinition*, TypeEnv)
    -> struct FunctionDefinition*;

auto TopLevel(std::list<Declaration>* fs) -> TypeCheckContext;

void PrintErrorString(const std::string& s);

}  // namespace Carbon

#endif  // EXECUTABLE_SEMANTICS_INTERPRETER_TYPECHECK_H_
