// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXECUTABLE_SEMANTICS_INTERP_H
#define EXECUTABLE_SEMANTICS_INTERP_H

#include <list>
#include <utility>
#include <vector>

#include "executable_semantics/assoc_list.h"
#include "executable_semantics/ast.h"
#include "executable_semantics/cons_list.h"

struct Value;

using Address = unsigned int;
using VarValues = std::list<std::pair<std::string, Value*>>;
using Env = AssocList<std::string, Address>;

/***** Values *****/

enum class ValKind {
  IntV,
  FunV,
  PtrV,
  BoolV,
  StructV,
  AltV,
  TupleV,
  VarTV,
  IntTV,
  BoolTV,
  TypeTV,
  FunctionTV,
  PointerTV,
  AutoTV,
  TupleTV,
  StructTV,
  ChoiceTV,
  VarPatV,
  AltConsV
};

struct Value {
  ValKind tag;
  bool alive;
  union {
    int integer;
    bool boolean;
    struct {
      std::string* name;
      Value* param;
      Statement* body;
    } fun;
    struct {
      Value* type;
      Value* inits;
    } struct_val;
    struct {
      std::string* alt_name;
      std::string* choice_name;
    } alt_cons;
    struct {
      std::string* alt_name;
      std::string* choice_name;
      Value* arg;
    } alt;
    struct {
      std::vector<std::pair<std::string, Address>>* elts;
    } tuple;
    Address ptr;
    std::string* var_type;
    struct {
      std::string* name;
      Value* type;
    } var_pat;
    struct {
      Value* param;
      Value* ret;
    } fun_type;
    struct {
      Value* type;
    } ptr_type;
    struct {
      std::string* name;
      VarValues* fields;
      VarValues* methods;
    } struct_type;
    struct {
      std::string* name;
      VarValues* fields;
    } tuple_type;
    struct {
      std::string* name;
      VarValues* alternatives;
    } choice_type;
    struct {
      std::list<std::string*>* params;
      Value* type;
    } implicit;
  } u;
};

auto MakeIntVal(int i) -> Value*;
auto MakeBoolVal(bool b) -> Value*;
auto MakeFunVal(std::string name, VarValues* implicit_params, Value* param,
                Statement* body, std::vector<Value*>* implicit_args) -> Value*;
auto MakePtrVal(Address addr) -> Value*;
auto MakeStructVal(Value* type,
                   std::vector<std::pair<std::string, Address>>* inits)
    -> Value*;
auto MakeTupleVal(std::vector<std::pair<std::string, Address>>* elts) -> Value*;
auto MakeAltVal(std::string alt_name, std::string choice_name, Value* arg)
    -> Value*;
auto MakeAltCons(std::string alt_name, std::string choice_name) -> Value*;

auto MakeVarPatVal(std::string name, Value* type) -> Value*;

auto MakeVarTypeVal(std::string name) -> Value*;
auto MakeIntTypeVal() -> Value*;
auto MakeAutoTypeVal() -> Value*;
auto MakeBoolTypeVal() -> Value*;
auto MakeTypeTypeVal() -> Value*;
auto MakeFunTypeVal(Value* param, Value* ret) -> Value*;
auto MakePtrTypeVal(Value* type) -> Value*;
auto MakeStructTypeVal(std::string name, VarValues* fields, VarValues* methods)
    -> Value*;
auto MakeTupleTypeVal(VarValues* fields) -> Value*;
auto MakeVoidTypeVal() -> Value*;
auto MakeChoiceTypeVal(std::string* name, VarValues* alts) -> Value*;

auto ValueEqual(Value* v1, Value* v2, int line_num) -> bool;

/***** Actions *****/

enum class ActionKind {
  LValAction,
  ExpressionAction,
  StatementAction,
  ValAction,
  ExpToLValAction,
  DeleteTmpAction
};

struct Action {
  ActionKind tag;
  union {
    Expression* exp;  // for LValAction and ExpressionAction
    Statement* stmt;
    Value* val;  // for finished actions with a value (ValAction)
    Address delete_tmp;
  } u;
  int pos;                      // position or state of the action
  std::vector<Value*> results;  // results from subexpression
};
using ActionList = Cons<Action*>*;

/***** Scopes *****/

struct Scope {
  Scope(Env* e, std::list<std::string> l) : env(e), locals(std::move(l)) {}
  Env* env;
  std::list<std::string> locals;
};

/***** Frames and State *****/

struct Frame {
  std::string name;
  Cons<Scope*>* scopes;
  Cons<Action*>* todo;

  Frame(std::string n, Cons<Scope*>* s, Cons<Action*>* c)
      : name(std::move(std::move(n))), scopes(s), todo(c) {}
};

struct State {
  Cons<Frame*>* stack;
  std::vector<Value*> heap;
};

extern State* state;

void PrintValue(Value* val, std::ostream& out);
void PrintEnv(Env* env);
auto AllocateValue(Value* v) -> Address;
auto CopyVal(Value* val, int line_num) -> Value*;
auto ToInteger(Value* v) -> int;

/***** Interpreters *****/

auto InterpProgram(std::list<Declaration*>* fs) -> int;
auto InterpExp(Env* env, Expression* e) -> Value*;

#endif  // EXECUTABLE_SEMANTICS_INTERP_H
