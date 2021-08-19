// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXECUTABLE_SEMANTICS_INTERPRETER_VALUE_H_
#define EXECUTABLE_SEMANTICS_INTERPRETER_VALUE_H_

#include <list>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "common/ostream.h"
#include "executable_semantics/ast/function_definition.h"
#include "executable_semantics/ast/statement.h"
#include "executable_semantics/common/ptr.h"
#include "executable_semantics/interpreter/address.h"
#include "executable_semantics/interpreter/field_path.h"
#include "executable_semantics/interpreter/stack.h"
#include "llvm/Support/Compiler.h"

namespace Carbon {

// Abstract base class of all AST nodes representing values.
//
// Value and its derived classes support LLVM-style RTTI, including
// llvm::isa, llvm::cast, and llvm::dyn_cast. To support this, every
// class derived from Value must provide a `classof` operation, and
// every concrete derived class must have a corresponding enumerator
// in `Kind`; see https://llvm.org/docs/HowToSetUpLLVMStyleRTTI.html for
// details.
class Value {
 public:
  enum class Kind {
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
    ClassType,
    ChoiceType,
    ContinuationType,  // The type of a continuation.
    VariableType,      // e.g., generic type parameters.
    BindingPlaceholderValue,
    AlternativeConstructorValue,
    ContinuationValue,  // A first-class continuation value.
    StringType,
    StringValue,
  };

  Value(const Value&) = delete;
  Value& operator=(const Value&) = delete;

  // Returns the enumerator corresponding to the most-derived type of this
  // object.
  auto Tag() const -> Kind { return tag; }

  void Print(llvm::raw_ostream& out) const;
  LLVM_DUMP_METHOD void Dump() const { Print(llvm::errs()); }

  // Returns the sub-Value specified by `path`, which must be a valid field
  // path for *this.
  auto GetField(const FieldPath& path, int line_num) const -> const Value*;

  // Returns a copy of *this, but with the sub-Value specified by `path`
  // set to `field_value`. `path` must be a valid field path for *this.
  auto SetField(const FieldPath& path, const Value* field_value,
                int line_num) const -> const Value*;

 protected:
  // Constructs a Value. `tag` must be the enumerator corresponding to the
  // most-derived type being constructed.
  explicit Value(Kind tag) : tag(tag) {}

 private:
  const Kind tag;
};

using VarValues = std::list<std::pair<std::string, const Value*>>;

auto FindInVarValues(const std::string& field, const VarValues& inits)
    -> const Value*;
auto FieldsEqual(const VarValues& ts1, const VarValues& ts2) -> bool;

// A TupleElement represents the value of a single tuple field.
struct TupleElement {
  // The field name.
  std::string name;

  // The field's value.
  const Value* value;
};

struct Frame;  // Used by continuation.

// An integer value.
class IntValue : public Value {
 public:
  explicit IntValue(int val) : Value(Kind::IntValue), val(val) {}

  static auto classof(const Value* value) -> bool {
    return value->Tag() == Kind::IntValue;
  }

  auto Val() const -> int { return val; }

 private:
  int val;
};

// A function value.
class FunctionValue : public Value {
 public:
  FunctionValue(std::string name, const Value* param, const Statement* body)
      : Value(Kind::FunctionValue),
        name(std::move(name)),
        param(param),
        body(body) {}

  static auto classof(const Value* value) -> bool {
    return value->Tag() == Kind::FunctionValue;
  }

  auto Name() const -> const std::string& { return name; }
  auto Param() const -> const Value* { return param; }
  auto Body() const -> const Statement* { return body; }

 private:
  std::string name;
  const Value* param;
  const Statement* body;
};

// A pointer value.
class PointerValue : public Value {
 public:
  explicit PointerValue(Address val)
      : Value(Kind::PointerValue), val(std::move(val)) {}

  static auto classof(const Value* value) -> bool {
    return value->Tag() == Kind::PointerValue;
  }

  auto Val() const -> const Address& { return val; }

 private:
  Address val;
};

// A bool value.
class BoolValue : public Value {
 public:
  explicit BoolValue(bool val) : Value(Kind::BoolValue), val(val) {}

  static auto classof(const Value* value) -> bool {
    return value->Tag() == Kind::BoolValue;
  }

  auto Val() const -> bool { return val; }

 private:
  bool val;
};

// A function value.
class StructValue : public Value {
 public:
  StructValue(const Value* type, const Value* inits)
      : Value(Kind::StructValue), type(type), inits(inits) {}

  static auto classof(const Value* value) -> bool {
    return value->Tag() == Kind::StructValue;
  }

  auto Type() const -> const Value* { return type; }
  auto Inits() const -> const Value* { return inits; }

 private:
  const Value* type;
  const Value* inits;
};

// An alternative constructor value.
class AlternativeConstructorValue : public Value {
 public:
  AlternativeConstructorValue(std::string alt_name, std::string choice_name)
      : Value(Kind::AlternativeConstructorValue),
        alt_name(std::move(alt_name)),
        choice_name(std::move(choice_name)) {}

  static auto classof(const Value* value) -> bool {
    return value->Tag() == Kind::AlternativeConstructorValue;
  }

  auto AltName() const -> const std::string& { return alt_name; }
  auto ChoiceName() const -> const std::string& { return choice_name; }

 private:
  std::string alt_name;
  std::string choice_name;
};

// An alternative value.
class AlternativeValue : public Value {
 public:
  AlternativeValue(std::string alt_name, std::string choice_name,
                   const Value* argument)
      : Value(Kind::AlternativeValue),
        alt_name(std::move(alt_name)),
        choice_name(std::move(choice_name)),
        argument(argument) {}

  static auto classof(const Value* value) -> bool {
    return value->Tag() == Kind::AlternativeValue;
  }

  auto AltName() const -> const std::string& { return alt_name; }
  auto ChoiceName() const -> const std::string& { return choice_name; }
  auto Argument() const -> const Value* { return argument; }

 private:
  std::string alt_name;
  std::string choice_name;
  const Value* argument;
};

// A function value.
class TupleValue : public Value {
 public:
  // An empty tuple, also known as the unit type.
  static const TupleValue& Empty() {
    static const TupleValue empty = TupleValue(std::vector<TupleElement>());
    return empty;
  }

  explicit TupleValue(std::vector<TupleElement> elements)
      : Value(Kind::TupleValue), elements(std::move(elements)) {}

  static auto classof(const Value* value) -> bool {
    return value->Tag() == Kind::TupleValue;
  }

  auto Elements() const -> const std::vector<TupleElement>& { return elements; }

  // Returns the value of the field named `name` in this tuple, or
  // null if there is no such field.
  auto FindField(const std::string& name) const -> const Value*;

 private:
  std::vector<TupleElement> elements;
};

// A binding placeholder value.
class BindingPlaceholderValue : public Value {
 public:
  // nullopt represents the `_` placeholder.
  BindingPlaceholderValue(std::optional<std::string> name, const Value* type)
      : Value(Kind::BindingPlaceholderValue),
        name(std::move(name)),
        type(type) {}

  static auto classof(const Value* value) -> bool {
    return value->Tag() == Kind::BindingPlaceholderValue;
  }

  auto Name() const -> const std::optional<std::string>& { return name; }
  auto Type() const -> const Value* { return type; }

 private:
  std::optional<std::string> name;
  const Value* type;
};

// The int type.
class IntType : public Value {
 public:
  IntType() : Value(Kind::IntType) {}

  static auto classof(const Value* value) -> bool {
    return value->Tag() == Kind::IntType;
  }
};

// The bool type.
class BoolType : public Value {
 public:
  BoolType() : Value(Kind::BoolType) {}

  static auto classof(const Value* value) -> bool {
    return value->Tag() == Kind::BoolType;
  }
};

// A type type.
class TypeType : public Value {
 public:
  TypeType() : Value(Kind::TypeType) {}

  static auto classof(const Value* value) -> bool {
    return value->Tag() == Kind::TypeType;
  }
};

// A function type.
class FunctionType : public Value {
 public:
  FunctionType(std::vector<GenericBinding> deduced, const Value* param,
               const Value* ret)
      : Value(Kind::FunctionType),
        deduced(std::move(deduced)),
        param(param),
        ret(ret) {}

  static auto classof(const Value* value) -> bool {
    return value->Tag() == Kind::FunctionType;
  }

  auto Deduced() const -> const std::vector<GenericBinding>& { return deduced; }
  auto Param() const -> const Value* { return param; }
  auto Ret() const -> const Value* { return ret; }

 private:
  std::vector<GenericBinding> deduced;
  const Value* param;
  const Value* ret;
};

// A pointer type.
class PointerType : public Value {
 public:
  explicit PointerType(const Value* type)
      : Value(Kind::PointerType), type(type) {}

  static auto classof(const Value* value) -> bool {
    return value->Tag() == Kind::PointerType;
  }

  auto Type() const -> const Value* { return type; }

 private:
  const Value* type;
};

// The `auto` type.
class AutoType : public Value {
 public:
  AutoType() : Value(Kind::AutoType) {}

  static auto classof(const Value* value) -> bool {
    return value->Tag() == Kind::AutoType;
  }
};

// A struct type.
class ClassType : public Value {
 public:
  ClassType(std::string name, VarValues fields, VarValues methods)
      : Value(Kind::ClassType),
        name(std::move(name)),
        fields(std::move(fields)),
        methods(std::move(methods)) {}

  static auto classof(const Value* value) -> bool {
    return value->Tag() == Kind::ClassType;
  }

  auto Name() const -> const std::string& { return name; }
  auto Fields() const -> const VarValues& { return fields; }
  auto Methods() const -> const VarValues& { return methods; }

 private:
  std::string name;
  VarValues fields;
  VarValues methods;
};

// A choice type.
class ChoiceType : public Value {
 public:
  ChoiceType(std::string name, VarValues alternatives)
      : Value(Kind::ChoiceType),
        name(std::move(name)),
        alternatives(std::move(alternatives)) {}

  static auto classof(const Value* value) -> bool {
    return value->Tag() == Kind::ChoiceType;
  }

  auto Name() const -> const std::string& { return name; }
  auto Alternatives() const -> const VarValues& { return alternatives; }

 private:
  std::string name;
  VarValues alternatives;
};

// A continuation type.
class ContinuationType : public Value {
 public:
  ContinuationType() : Value(Kind::ContinuationType) {}

  static auto classof(const Value* value) -> bool {
    return value->Tag() == Kind::ContinuationType;
  }
};

// A variable type.
class VariableType : public Value {
 public:
  explicit VariableType(std::string name)
      : Value(Kind::VariableType), name(std::move(name)) {}

  static auto classof(const Value* value) -> bool {
    return value->Tag() == Kind::VariableType;
  }

  auto Name() const -> const std::string& { return name; }

 private:
  std::string name;
};

// A first-class continuation representation of a fragment of the stack.
class ContinuationValue : public Value {
 public:
  explicit ContinuationValue(std::vector<Ptr<Frame>> stack)
      : Value(Kind::ContinuationValue), stack(std::move(stack)) {}

  static auto classof(const Value* value) -> bool {
    return value->Tag() == Kind::ContinuationValue;
  }

  auto Stack() const -> const std::vector<Ptr<Frame>>& { return stack; }

 private:
  std::vector<Ptr<Frame>> stack;
};

// The String type.
class StringType : public Value {
 public:
  StringType() : Value(Kind::StringType) {}

  static auto classof(const Value* value) -> bool {
    return value->Tag() == Kind::StringType;
  }
};

// A string value.
class StringValue : public Value {
 public:
  explicit StringValue(std::string val)
      : Value(Kind::StringValue), val(std::move(val)) {}

  static auto classof(const Value* value) -> bool {
    return value->Tag() == Kind::StringValue;
  }

  auto Val() const -> const std::string& { return val; }

 private:
  std::string val;
};

auto CopyVal(const Value* val, int line_num) -> const Value*;

auto TypeEqual(const Value* t1, const Value* t2) -> bool;
auto ValueEqual(const Value* v1, const Value* v2, int line_num) -> bool;

}  // namespace Carbon

#endif  // EXECUTABLE_SEMANTICS_INTERPRETER_VALUE_H_
