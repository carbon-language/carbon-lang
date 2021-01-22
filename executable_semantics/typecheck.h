// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef TYPECHECK_H
#define TYPECHECK_H

#include <set>
using std::pair;
using std::set;
#include "executable_semantics/assoc_list.h"
#include "executable_semantics/ast.h"
#include "executable_semantics/interp.h"

typedef AList<string, Value*> TypeEnv;

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

Value* to_type(int lineno, Value* val);

bool type_equal(Value* t1, Value* t2);
bool fields_equal(VarValues* ts1, VarValues* ts2);

TCResult typecheck_exp(Expression* e, TypeEnv* env, Env* ct_env,
                       Value* expected, TCContext context);

TCStatement typecheck_stmt(Statement*, TypeEnv*, Env*, Value*);

struct FunctionDefinition* typecheck_fun_def(struct FunctionDefinition*,
                                             TypeEnv*);

Declaration* typecheck_decl(Declaration* d, TypeEnv* env, Env* ct_env);

pair<TypeEnv*, Env*> top_level(list<struct Declaration*>* fs);

void print_error_string(string s);

#endif
