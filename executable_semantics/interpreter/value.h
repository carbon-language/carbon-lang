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

struct Function {
  std::string* name;
  const Value* param;
  const Statement* body;
};

struct StructConstructor {
  const Value* type;
  const Value* inits;
};

struct AlternativeConstructor {
  std::string* alt_name;
  std::string* choice_name;
};

struct Alternative {
  std::string* alt_name;
  std::string* choice_name;
  Address argument;
};

struct TupleValue {
  std::vector<TupleElement>* elements;
};

struct VariablePatternValue {
  std::string* name;
  const Value* type;
};

struct FunctionTypeValue {
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
  static auto MakeContinuation(std::vector<Frame*> stack) -> Value*;
  static auto MakeIntVal(int i) -> const Value*;
  static auto MakeBoolVal(bool b) -> const Value*;
  static auto MakeFunVal(std::string name, const Value* param,
                         const Statement* body) -> const Value*;
  static auto MakePtrVal(Address addr) -> const Value*;
  static auto MakeStructVal(const Value* type, const Value* inits)
      -> const Value*;
  static auto MakeTupleVal(std::vector<TupleElement>* elts) -> const Value*;
  static auto MakeAltVal(std::string alt_name, std::string choice_name,
                         Address argument) -> const Value*;
  static auto MakeAltCons(std::string alt_name, std::string choice_name)
      -> const Value*;
  static auto MakeVarPatVal(std::string name, const Value* type)
      -> const Value*;
  static auto MakeVarTypeVal(std::string name) -> const Value*;
  static auto MakeIntTypeVal() -> const Value*;
  static auto MakeContinuationTypeVal() -> const Value*;
  static auto MakeAutoTypeVal() -> const Value*;
  static auto MakeBoolTypeVal() -> const Value*;
  static auto MakeTypeTypeVal() -> const Value*;
  static auto MakeFunTypeVal(const Value* param, const Value* ret)
      -> const Value*;
  static auto MakePtrTypeVal(const Value* type) -> const Value*;
  static auto MakeStructTypeVal(std::string name, VarValues* fields,
                                VarValues* methods) -> const Value*;
  static auto MakeUnitTypeVal() -> const Value*;
  static auto MakeChoiceTypeVal(std::string name, VarValues* alts)
      -> const Value*;

  // Access to alternatives
  int GetInteger() const;
  bool GetBoolean() const;
  Function GetFunction() const;
  StructConstructor GetStruct() const;
  AlternativeConstructor GetAlternativeConstructor() const;
  Alternative GetAlternative() const;
  TupleValue GetTuple() const;
  Address GetPointer() const;
  std::string* GetVariableType() const;
  VariablePatternValue GetVariablePattern() const;
  FunctionTypeValue GetFunctionType() const;
  PointerType GetPointerType() const;
  StructType GetStructType() const;
  ChoiceType GetChoiceType() const;
  ContinuationValue GetContinuation() const;

 private:
  union {
    int integer;
    bool boolean;
    Function fun;
    StructConstructor struct_val;
    AlternativeConstructor alt_cons;
    Alternative alt;
    TupleValue tuple;
    Address ptr;
    std::string* var_type;
    VariablePatternValue var_pat;
    FunctionTypeValue fun_type;
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
