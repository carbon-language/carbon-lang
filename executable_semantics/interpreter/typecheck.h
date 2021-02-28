// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXECUTABLE_SEMANTICS_INTERPRETER_TYPECHECK_H_
#define EXECUTABLE_SEMANTICS_INTERPRETER_TYPECHECK_H_

#include <set>

#include "executable_semantics/ast/expression.h"
#include "executable_semantics/ast/statement.h"
#include "executable_semantics/interpreter/assoc_list.h"
#include "executable_semantics/interpreter/interpreter.h"

namespace Carbon {

using TypeEnv = AssocList<std::string, Value*>;

void PrintTypeEnv(TypeEnv env);

enum class TCContext { ValueContext, PatternContext, TypeContext };

struct TCResult {
  TCResult(Expression* e, Value* t, TypeEnv env) : exp(e), type(t), env(env) {}

  Expression* exp;
  Value* type;
  TypeEnv env;
};

struct TCStatement {
  TCStatement(Statement* s, TypeEnv e) : stmt(s), env(e) {}

  Statement* stmt;
  TypeEnv env;
};

auto ToType(int line_num, Value* val) -> Value*;

auto TypeCheckExp(Expression* e, TypeEnv env, Env ct_env, Value* expected,
                  TCContext context) -> TCResult;

auto TypeCheckStmt(Statement*, TypeEnv, Env, Value*) -> TCStatement;

auto TypeCheckFunDef(struct FunctionDefinition*, TypeEnv)
    -> struct FunctionDefinition*;

auto TypeCheckDecl(Declaration* d, TypeEnv env, Env ct_env) -> Declaration*;

auto TopLevel(std::list<Declaration*>* fs) -> std::pair<TypeEnv, Env>;

void PrintErrorString(const std::string& s);

}  // namespace Carbon

#endif  // EXECUTABLE_SEMANTICS_INTERPRETER_TYPECHECK_H_
