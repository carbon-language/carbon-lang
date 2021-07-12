// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXECUTABLE_SEMANTICS_INTERPRETER_VALUE_H_
#define EXECUTABLE_SEMANTICS_INTERPRETER_VALUE_H_

#include <list>
#include <optional>
#include <variant>
#include <vector>

#include "executable_semantics/ast/statement.h"
#include "executable_semantics/interpreter/stack.h"

namespace Carbon {

struct Value;
using Address = unsigned int;
using VarValues = std::list<std::pair<std::string, const Value*>>;

auto FindInVarValues(const std::string& field, const VarValues& inits)
    -> const Value*;
auto FieldsEqual(const VarValues& ts1, const VarValues& ts2) -> bool;

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

struct IntValue {
  static constexpr ValKind Kind = ValKind::IntValue;
  int value;
};

struct FunctionValue {
  static constexpr ValKind Kind = ValKind::FunctionValue;
  std::string name;
  const Value* param;
  const Statement* body;
};

struct PointerValue {
  static constexpr ValKind Kind = ValKind::PointerValue;
  Address value;
};

struct BoolValue {
  static constexpr ValKind Kind = ValKind::BoolValue;
  bool value;
};

struct StructValue {
  static constexpr ValKind Kind = ValKind::StructValue;
  const Value* type;
  const Value* inits;
};

struct AlternativeConstructorValue {
  static constexpr ValKind Kind = ValKind::AlternativeConstructorValue;
  std::string alt_name;
  std::string choice_name;
};

struct AlternativeValue {
  static constexpr ValKind Kind = ValKind::AlternativeValue;
  std::string alt_name;
  std::string choice_name;
  Address argument;
};

struct TupleValue {
  static constexpr ValKind Kind = ValKind::TupleValue;
  std::vector<TupleElement> elements;
};

struct BindingPlaceholderValue {
  static constexpr ValKind Kind = ValKind::BindingPlaceholderValue;
  std::string name;
  const Value* type;
};

struct IntType {
  static constexpr ValKind Kind = ValKind::IntType;
};

struct BoolType {
  static constexpr ValKind Kind = ValKind::BoolType;
};

struct TypeType {
  static constexpr ValKind Kind = ValKind::TypeType;
};

struct FunctionType {
  static constexpr ValKind Kind = ValKind::FunctionType;
  const Value* param;
  const Value* ret;
};

struct PointerType {
  static constexpr ValKind Kind = ValKind::PointerType;
  const Value* type;
};

struct AutoType {
  static constexpr ValKind Kind = ValKind::AutoType;
};

struct StructType {
  static constexpr ValKind Kind = ValKind::StructType;
  std::string name;
  VarValues fields;
  VarValues methods;
};

struct ChoiceType {
  static constexpr ValKind Kind = ValKind::ChoiceType;
  std::string name;
  VarValues alternatives;
};

struct ContinuationType {
  static constexpr ValKind Kind = ValKind::ContinuationType;
};

struct ContinuationValue {
  static constexpr ValKind Kind = ValKind::ContinuationValue;
  std::vector<Frame*> stack;
};

struct Value {
  auto tag() const -> ValKind;

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
  static auto MakeTupleValue(std::vector<TupleElement> elts) -> const Value*;
  static auto MakeAlternativeValue(std::string alt_name,
                                   std::string choice_name, Address argument)
      -> const Value*;
  static auto MakeAlternativeConstructorValue(std::string alt_name,
                                              std::string choice_name)
      -> const Value*;
  static auto MakeBindingPlaceholderValue(std::string name, const Value* type)
      -> const Value*;
  static auto MakeIntType() -> const Value*;
  static auto MakeContinuationType() -> const Value*;
  static auto MakeAutoType() -> const Value*;
  static auto MakeBoolType() -> const Value*;
  static auto MakeTypeType() -> const Value*;
  static auto MakeFunctionType(const Value* param, const Value* ret)
      -> const Value*;
  static auto MakePointerType(const Value* type) -> const Value*;
  static auto MakeStructType(std::string name, VarValues fields,
                             VarValues methods) -> const Value*;
  static auto MakeUnitTypeVal() -> const Value*;
  static auto MakeChoiceType(std::string name, VarValues alts) -> const Value*;

  // Access to alternatives
  auto GetIntValue() const -> int;
  auto GetBoolValue() const -> bool;
  auto GetFunctionValue() const -> const FunctionValue&;
  auto GetStructValue() const -> const StructValue&;
  auto GetAlternativeConstructorValue() const
      -> const AlternativeConstructorValue&;
  auto GetAlternativeValue() const -> const AlternativeValue&;
  auto GetTupleValue() const -> const TupleValue&;
  auto GetPointerValue() const -> Address;
  auto GetBindingPlaceholderValue() const -> const BindingPlaceholderValue&;
  auto GetFunctionType() const -> const FunctionType&;
  auto GetPointerType() const -> const PointerType&;
  auto GetStructType() const -> const StructType&;
  auto GetChoiceType() const -> const ChoiceType&;
  auto GetContinuationValue() const -> const ContinuationValue&;

 private:
  std::variant<IntValue, FunctionValue, PointerValue, BoolValue, StructValue,
               AlternativeValue, TupleValue, IntType, BoolType, TypeType,
               FunctionType, PointerType, AutoType, StructType, ChoiceType,
               ContinuationType, BindingPlaceholderValue,
               AlternativeConstructorValue, ContinuationValue>
      value;
};

void PrintValue(const Value* val, std::ostream& out);

auto TypeEqual(const Value* t1, const Value* t2) -> bool;
auto ValueEqual(const Value* v1, const Value* v2, int line_num) -> bool;

auto ToInteger(const Value* v) -> int;

}  // namespace Carbon

#endif  // EXECUTABLE_SEMANTICS_INTERPRETER_VALUE_H_
