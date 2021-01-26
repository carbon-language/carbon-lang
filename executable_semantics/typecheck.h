// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXECUTABLE_SEMANTICS_TYPECHECK_H
#define EXECUTABLE_SEMANTICS_TYPECHECK_H

#include <set>

#include "executable_semantics/assoc_list.h"
#include "executable_semantics/ast.h"
#include "executable_semantics/interp.h"

typedef AList<std::string, Value*> TypeEnv;

void print_type_env(TypeEnv* env);

enum TCContext { ValueContext, PatternContext, TypeContext };

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

Value* ToType(int lineno, Value* val);

bool TypeEqual(Value* t1, Value* t2);
bool FieldsEqual(VarValues* ts1, VarValues* ts2);

TCResult TypeCheckExp(Expression* e, TypeEnv* env, Env* ct_env, Value* expected,
                      TCContext context);

TCStatement TypeCheckStmt(Statement*, TypeEnv*, Env*, Value*);

struct FunctionDefinition* TypeCheckFunDef(struct FunctionDefinition*,
                                           TypeEnv*);

Declaration* TypeCheckDecl(Declaration* d, TypeEnv* env, Env* ct_env);

std::pair<TypeEnv*, Env*> TopLevel(std::list<struct Declaration*>* fs);

void PrintErrorString(std::string s);

#endif
