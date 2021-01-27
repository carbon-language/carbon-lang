// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXECUTABLE_SEMANTICS_TYPECHECK_H
#define EXECUTABLE_SEMANTICS_TYPECHECK_H

#include <set>

#include "executable_semantics/assoc_list.h"
#include "executable_semantics/ast.h"
#include "executable_semantics/interp.h"

using TypeEnv = AssocList<std::string, Value*>;

void PrintTypeEnv(TypeEnv* env);

enum class TCContext { ValueContext, PatternContext, TypeContext };

struct TCResult {
  TCResult(Expression* e, Value* t, TypeEnv* env) : exp(e), type(t), env(env) {}

  Expression* exp;
  Value* type;
  TypeEnv* env;
};

struct TCStatement {
  TCStatement(Statement* s, TypeEnv* e) : stmt(s), env(e) {}

  Statement* stmt;
  TypeEnv* env;
};

auto ToType(int line_num, Value* val) -> Value*;

auto TypeEqual(Value* t1, Value* t2) -> bool;
auto FieldsEqual(VarValues* ts1, VarValues* ts2) -> bool;

auto TypeCheckExp(Expression* e, TypeEnv* env, Env* ct_env, Value* expected,
                  TCContext context) -> TCResult;

auto TypeCheckStmt(Statement*, TypeEnv*, Env*, Value*) -> TCStatement;

auto TypeCheckFunDef(struct FunctionDefinition*, TypeEnv*)
    -> struct FunctionDefinition*;

auto TypeCheckDecl(Declaration* d, TypeEnv* env, Env* ct_env) -> Declaration*;

auto TopLevel(std::list<struct Declaration*>* fs) -> std::pair<TypeEnv*, Env*>;

void PrintErrorString(const std::string& s);

#endif  // EXECUTABLE_SEMANTICS_TYPECHECK_H
