// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef INTERP_H
#define INTERP_H

#include <list>
#include <vector>

#include "executable_semantics/assoc_list.h"
#include "executable_semantics/ast.h"
#include "executable_semantics/cons_list.h"

struct Value;

typedef unsigned int Address;
typedef std::list<std::pair<std::string, Value*> > VarValues;
typedef AList<std::string, Address> Env;

/***** Values *****/

enum ValKind {
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
      std::vector<std::pair<std::string, Address> >* elts;
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

Value* MakeInt_val(int i);
Value* MakeBool_val(bool b);
Value* MakeFun_val(std::string name, VarValues* implicit_params, Value* param,
                   Statement* body, std::vector<Value*>* implicit_args);
Value* MakePtr_val(Address addr);
Value* MakeStruct_val(Value* type,
                      std::vector<std::pair<std::string, Address> >* inits);
Value* MakeTuple_val(std::vector<std::pair<std::string, Address> >* elts);
Value* MakeAlt_val(std::string alt_name, std::string choice_name, Value* arg);
Value* MakeAlt_cons(std::string alt_name, std::string choice_name);

Value* MakeVarPat_val(std::string name, Value* type);

Value* MakeVar_type_val(std::string name);
Value* MakeIntType_val();
Value* MakeAutoType_val();
Value* MakeBoolType_val();
Value* MakeTypeType_val();
Value* MakeFunType_val(Value* param, Value* ret);
Value* MakePtr_type_val(Value* type);
Value* MakeStruct_type_val(std::string name, VarValues* fields,
                           VarValues* methods);
Value* MakeTuple_type_val(VarValues* fields);
Value* MakeVoid_type_val();
Value* MakeChoice_type_val(std::string* name, VarValues* alts);

bool value_equal(Value* v1, Value* v2, int lineno);

/***** Actions *****/

enum ActionKind {
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
    Address delete_;
  } u;
  int pos;                 // position or state of the action
  std::vector<Value*> results;  // results from subexpression
};
typedef Cons<Action*>* ActionList;

/***** Scopes *****/

struct Scope {
  Scope(Env* e, const std::list<std::string>& l) : env(e), locals(l) {}
  Env* env;
  std::list<std::string> locals;
};

/***** Frames and State *****/

struct Frame {
  std::string name;
  Cons<Scope*>* scopes;
  Cons<Action*>* todo;

  Frame(std::string n, Cons<Scope*>* s, Cons<Action*>* c)
      : name(n), scopes(s), todo(c) {}
};

struct State {
  Cons<Frame*>* stack;
  std::vector<Value*> heap;
};

extern State* state;

void print_value(Value* val, std::ostream& out);
void print_env(Env* env);
Address allocate_value(Value* v);
Value* copy_val(Value* val, int lineno);
int to_integer(Value* v);

/***** Interpreters *****/

int interp_program(std::list<Declaration*>* fs);
Value* interp_exp(Env* env, Expression* e);

#endif
