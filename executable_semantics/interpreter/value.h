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
using VarAddresses = std::vector<std::pair<std::string, Address>>;

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
  VarAddresses* elts;
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

  // TODO: replace these constructors functions with real constructors
  //
  // RANT: The following long list of friend declarations is an
  // example of a problem in the design of C++. It is so focused on
  // classes and objects that it fails for modular procedural
  // programming. There are better ways to control access, for
  // example, going back to the module system of in CLU programming
  // language in the 1970's. -Jeremy

  friend auto MakeContinuation(std::vector<Frame*> stack) -> Value*;
  friend auto MakeIntVal(int i) -> const Value*;
  friend auto MakeBoolVal(bool b) -> const Value*;
  friend auto MakeFunVal(std::string name, const Value* param,
                         const Statement* body) -> const Value*;
  friend auto MakePtrVal(Address addr) -> const Value*;
  friend auto MakeStructVal(const Value* type, const Value* inits)
      -> const Value*;
  friend auto MakeTupleVal(std::vector<std::pair<std::string, Address>>* elts)
      -> const Value*;
  friend auto MakeAltVal(std::string alt_name, std::string choice_name,
                         Address argument) -> const Value*;
  friend auto MakeAltCons(std::string alt_name, std::string choice_name)
      -> const Value*;

  friend auto MakeVarPatVal(std::string name, const Value* type)
      -> const Value*;

  friend auto MakeVarTypeVal(std::string name) -> const Value*;
  friend auto MakeIntTypeVal() -> const Value*;
  friend auto MakeContinuationTypeVal() -> const Value*;
  friend auto MakeAutoTypeVal() -> const Value*;
  friend auto MakeBoolTypeVal() -> const Value*;
  friend auto MakeTypeTypeVal() -> const Value*;
  friend auto MakeFunTypeVal(const Value* param, const Value* ret)
      -> const Value*;
  friend auto MakePtrTypeVal(const Value* type) -> const Value*;
  friend auto MakeStructTypeVal(std::string name, VarValues* fields,
                                VarValues* methods) -> const Value*;
  friend auto MakeVoidTypeVal() -> const Value*;
  friend auto MakeChoiceTypeVal(std::string name, VarValues* alts)
      -> const Value*;
};

// Return a first-class continuation represented by the
// given stack, down to the nearest enclosing `__continuation`.
auto MakeContinuation(std::vector<Frame*> stack) -> Value*;
auto MakeIntVal(int i) -> const Value*;
auto MakeBoolVal(bool b) -> const Value*;
auto MakeFunVal(std::string name, const Value* param, const Statement* body)
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
