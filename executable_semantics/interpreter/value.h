// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXECUTABLE_SEMANTICS_INTERPRETER_VALUE_H_
#define EXECUTABLE_SEMANTICS_INTERPRETER_VALUE_H_

#include <list>
#include <optional>
#include <vector>

#include "executable_semantics/ast/statement.h"
#include "executable_semantics/interpreter/stack.h"

namespace Carbon {

struct Value;
using Address = unsigned int;
using VarValues = std::list<std::pair<std::string, const Value*>>;

auto FindInVarValues(const std::string& field, VarValues* inits)
    -> const Value*;
auto FieldsEqual(VarValues* ts1, VarValues* ts2) -> bool;

// Finds the field in `*tuple` named `name`, and returns its address, or
// nullopt if there is no such field. `*tuple` must be a tuple value.
auto FindTupleField(const std::string& name, const Value* tuple)
    -> std::optional<Address>;

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
  StructTV,
  ChoiceTV,
  ContinuationTV,  // The type of a continuation.
  VarPatV,
  AltConsV,
  ContinuationV  // A first-class continuation value.
};

struct Frame;  // used by continuation

struct Value {
  ValKind tag;
  union {
    int integer;
    bool boolean;

    struct {
      std::string* name;
      const Value* param;
      Statement* body;
    } fun;

    struct {
      const Value* type;
      const Value* inits;
    } struct_val;

    struct {
      std::string* alt_name;
      std::string* choice_name;
    } alt_cons;

    struct {
      std::string* alt_name;
      std::string* choice_name;
      Address argument;
    } alt;

    struct {
      std::vector<std::pair<std::string, Address>>* elts;
    } tuple;

    Address ptr;
    std::string* var_type;

    struct {
      std::string* name;
      const Value* type;
    } var_pat;

    struct {
      const Value* param;
      const Value* ret;
    } fun_type;

    struct {
      const Value* type;
    } ptr_type;

    struct {
      std::string* name;
      VarValues* fields;
      VarValues* methods;
    } struct_type;

    struct {
      std::string* name;
      VarValues* alternatives;
    } choice_type;

    struct {
      std::list<std::string*>* params;
      const Value* type;
    } implicit;

    struct {
      std::vector<Frame*>* stack;
    } continuation;

  } u;
};

// Return a first-class continuation represented by the
// given stack, down to the nearest enclosing `__continuation`.
auto MakeContinuation(std::vector<Frame*> stack) -> Value*;
auto MakeIntVal(int i) -> const Value*;
auto MakeBoolVal(bool b) -> const Value*;
auto MakeFunVal(std::string name, const Value* param, Statement* body)
    -> const Value*;
auto MakePtrVal(Address addr) -> const Value*;
auto MakeStructVal(const Value* type, const Value* inits) -> const Value*;
auto MakeTupleVal(std::vector<std::pair<std::string, Address>>* elts)
    -> const Value*;
auto MakeAltVal(std::string alt_name, std::string choice_name, Address argument)
    -> const Value*;
auto MakeAltCons(std::string alt_name, std::string choice_name) -> const Value*;

auto MakeVarPatVal(std::string name, const Value* type) -> const Value*;

auto MakeVarTypeVal(std::string name) -> const Value*;
auto MakeIntTypeVal() -> const Value*;
auto MakeContinuationTypeVal() -> const Value*;
auto MakeAutoTypeVal() -> const Value*;
auto MakeBoolTypeVal() -> const Value*;
auto MakeTypeTypeVal() -> const Value*;
auto MakeFunTypeVal(const Value* param, const Value* ret) -> const Value*;
auto MakePtrTypeVal(const Value* type) -> const Value*;
auto MakeStructTypeVal(std::string name, VarValues* fields, VarValues* methods)
    -> const Value*;
auto MakeVoidTypeVal() -> const Value*;
auto MakeChoiceTypeVal(std::string name, VarValues* alts) -> const Value*;

void PrintValue(const Value* val, std::ostream& out);

auto TypeEqual(const Value* t1, const Value* t2) -> bool;
auto ValueEqual(const Value* v1, const Value* v2, int line_num) -> bool;

auto ToInteger(const Value* v) -> int;

}  // namespace Carbon

#endif  // EXECUTABLE_SEMANTICS_INTERPRETER_VALUE_H_
