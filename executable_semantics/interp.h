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

using std::list;
using std::vector;

struct Value;

typedef unsigned int address;
typedef list<pair<string, Value*> > VarValues;
typedef AList<string, address> Env;

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
      string* name;
      Value* param;
      Statement* body;
    } fun;
    struct {
      Value* type;
      Value* inits;
    } struct_val;
    struct {
      string* alt_name;
      string* choice_name;
    } alt_cons;
    struct {
      string* alt_name;
      string* choice_name;
      Value* arg;
    } alt;
    struct {
      vector<pair<string, address> >* elts;
    } tuple;
    address ptr;
    string* var_type;
    struct {
      string* name;
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
      string* name;
      VarValues* fields;
      VarValues* methods;
    } struct_type;
    struct {
      string* name;
      VarValues* fields;
    } tuple_type;
    struct {
      string* name;
      VarValues* alternatives;
    } choice_type;
    struct {
      list<string*>* params;
      Value* type;
    } implicit;
  } u;
};

Value* MakeInt_val(int i);
Value* MakeBool_val(bool b);
Value* MakeFun_val(string name, VarValues* implicit_params, Value* param,
                    Statement* body, vector<Value*>* implicit_args);
Value* MakePtr_val(address addr);
Value* MakeStruct_val(Value* type, vector<pair<string, address> >* inits);
Value* MakeTuple_val(vector<pair<string, address> >* elts);
Value* MakeAlt_val(string alt_name, string choice_name, Value* arg);
Value* MakeAlt_cons(string alt_name, string choice_name);

Value* MakeVarPat_val(string name, Value* type);

Value* MakeVar_type_val(string name);
Value* MakeIntType_val();
Value* MakeAutoType_val();
Value* MakeBoolType_val();
Value* MakeTypeType_val();
Value* MakeFunType_val(Value* param, Value* ret);
Value* MakePtr_type_val(Value* type);
Value* MakeStruct_type_val(string name, VarValues* fields, VarValues* methods);
Value* MakeTuple_type_val(VarValues* fields);
Value* MakeVoid_type_val();
Value* MakeChoice_type_val(string* name, VarValues* alts);

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
    address delete_;
  } u;
  int pos;                 // position or state of the action
  vector<Value*> results;  // results from subexpression
};
typedef Cons<Action*>* ActionList;

/***** Scopes *****/

struct Scope {
  Scope(Env* e, const list<string>& l) : env(e), locals(l) {}
  Env* env;
  list<string> locals;
};

/***** Frames and State *****/

struct Frame {
  string name;
  Cons<Scope*>* scopes;
  Cons<Action*>* todo;

  Frame(string n, Cons<Scope*>* s, Cons<Action*>* c)
      : name(n), scopes(s), todo(c) {}
};

struct State {
  Cons<Frame*>* stack;
  vector<Value*> heap;
};

extern State* state;

void print_value(Value* val, std::ostream& out);
void print_env(Env* env);
address allocate_value(Value* v);
Value* copy_val(Value* val, int lineno);
int to_integer(Value* v);

/***** Interpreters *****/

int interp_program(list<Declaration*>* fs);
Value* interp_exp(Env* env, Expression* e);

#endif
