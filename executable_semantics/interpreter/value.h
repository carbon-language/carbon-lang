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

// A TupleElement represents the value of a single tuple field.
struct TupleElement {
  // The field name.
  std::string name;

  // Location of the field's value.
  Address address;
};

enum class ValKind {
  IntValue,
  FunctionValue,
  PointerValue,
  BoolValue,
  StructValue,
  AlternativeValue,
  TupleValue,
  VarTV,
  IntType,
  BoolType,
  TypeType,
  FunctionType,
  PointerType,
  AutoType,
  StructType,
  ChoiceType,
  ContinuationType,  // The type of a continuation.
  BindingPlaceholderValue,
  AlternativeConstructorValue,
  ContinuationValue  // A first-class continuation value.
};

struct Frame;  // used by continuation

struct FunctionValue {
  std::string* name;
  const Value* param;
  const Statement* body;
};

struct StructValue {
  const Value* type;
  const Value* inits;
};

struct AlternativeConstructorValue {
  std::string* alt_name;
  std::string* choice_name;
};

struct AlternativeValue {
  std::string* alt_name;
  std::string* choice_name;
  Address argument;
};

struct TupleValue {
  std::vector<TupleElement>* elements;
};

struct BindingPlaceholderValue {
  std::string* name;
  const Value* type;
};

struct FunctionType {
  const Value* param;
  const Value* ret;
};

struct PointerType {
  const Value* type;
};

struct StructType {
  std::string* name;
  VarValues* fields;
  VarValues* methods;
};

struct ChoiceType {
  std::string* name;
  VarValues* alternatives;
};

struct ContinuationValue {
  std::vector<Frame*>* stack;
};

struct Value {
  ValKind tag;

  // Constructors

  // Return a first-class continuation represented by the
  // given stack, down to the nearest enclosing `__continuation`.
  static auto MakeContinuationValue(std::vector<Frame*> stack) -> Value*;
  static auto MakeIntValue(int i) -> const Value*;
  static auto MakeBoolValue(bool b) -> const Value*;
  static auto MakeFunctionValue(std::string name, const Value* param,
                                const Statement* body) -> const Value*;
  static auto MakePointerValue(Address addr) -> const Value*;
  static auto MakeStructValue(const Value* type, const Value* inits)
      -> const Value*;
  static auto MakeTupleValue(std::vector<TupleElement>* elts) -> const Value*;
  static auto MakeAlternativeValue(std::string alt_name,
                                   std::string choice_name, Address argument)
      -> const Value*;
  static auto MakeAlternativeConstructorValue(std::string alt_name,
                                              std::string choice_name)
      -> const Value*;
  static auto MakeBindingPlaceholderValue(std::string name, const Value* type)
      -> const Value*;
  static auto MakeVarTypeVal(std::string name) -> const Value*;
  static auto MakeIntType() -> const Value*;
  static auto MakeContinuationType() -> const Value*;
  static auto MakeAutoType() -> const Value*;
  static auto MakeBoolType() -> const Value*;
  static auto MakeTypeType() -> const Value*;
  static auto MakeFunctionType(const Value* param, const Value* ret)
      -> const Value*;
  static auto MakePointerType(const Value* type) -> const Value*;
  static auto MakeStructType(std::string name, VarValues* fields,
                             VarValues* methods) -> const Value*;
  static auto MakeUnitTypeVal() -> const Value*;
  static auto MakeChoiceType(std::string name, VarValues* alts) -> const Value*;

  // Access to alternatives
  int GetIntValue() const;
  bool GetBoolValue() const;
  FunctionValue GetFunctionValue() const;
  StructValue GetStructValue() const;
  AlternativeConstructorValue GetAlternativeConstructorValue() const;
  AlternativeValue GetAlternativeValue() const;
  TupleValue GetTupleValue() const;
  Address GetPointerValue() const;
  std::string* GetVariableType() const;
  BindingPlaceholderValue GetBindingPlaceholderValue() const;
  FunctionType GetFunctionType() const;
  PointerType GetPointerType() const;
  StructType GetStructType() const;
  ChoiceType GetChoiceType() const;
  ContinuationValue GetContinuationValue() const;

 private:
  union {
    int integer;
    bool boolean;
    FunctionValue fun;
    StructValue struct_val;
    AlternativeConstructorValue alt_cons;
    AlternativeValue alt;
    TupleValue tuple;
    Address ptr;
    std::string* var_type;
    BindingPlaceholderValue var_pat;
    FunctionType fun_type;
    PointerType ptr_type;
    StructType struct_type;
    ChoiceType choice_type;
    ContinuationValue continuation;
  } u;
};

void PrintValue(const Value* val, std::ostream& out);

auto TypeEqual(const Value* t1, const Value* t2) -> bool;
auto ValueEqual(const Value* v1, const Value* v2, int line_num) -> bool;

auto ToInteger(const Value* v) -> int;

}  // namespace Carbon

#endif  // EXECUTABLE_SEMANTICS_INTERPRETER_VALUE_H_
