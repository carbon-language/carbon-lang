// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXECUTABLE_SEMANTICS_INTERPRETER_VALUE_H_
#define EXECUTABLE_SEMANTICS_INTERPRETER_VALUE_H_

#include <iostream>
#include <list>
#include <vector>

#include "executable_semantics/ast/statement.h"

namespace Carbon {

struct Value;
using Address = unsigned int;
using VarValues = std::list<std::pair<std::string, Value*>>;

auto FindInVarValues(const std::string& field, VarValues* inits) -> Value*;
auto FieldsEqual(VarValues* ts1, VarValues* ts2) -> bool;

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
auto MakeFunVal(std::string name, Value* param, Statement* body) -> Value*;
auto MakePtrVal(Address addr) -> Value*;
auto MakeStructVal(Value* type, Value* inits) -> Value*;
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

void PrintValue(Value* val, std::ostream& out);
// The following will be possible once Value is a value type! -Jeremy
// auto operator<<(std::ostream& os, Value* v) -> std::ostream&;

auto TypeEqual(Value* t1, Value* t2) -> bool;
auto ValueEqual(Value* v1, Value* v2, int line_num) -> bool;

auto ToInteger(Value* v) -> int;
void CheckAlive(Value* v, int line_num);

}  // namespace Carbon

#endif  // EXECUTABLE_SEMANTICS_INTERPRETER_VALUE_H_
