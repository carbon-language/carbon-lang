// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXECUTABLE_SEMANTICS_INTERPRETER_VALUE_H_
#define EXECUTABLE_SEMANTICS_INTERPRETER_VALUE_H_

#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "common/ostream.h"
#include "executable_semantics/ast/function_definition.h"
#include "executable_semantics/ast/statement.h"
#include "executable_semantics/common/nonnull.h"
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
    NominalClassValue,
    AlternativeValue,
    TupleValue,
    IntType,
    BoolType,
    TypeType,
    FunctionType,
    PointerType,
    AutoType,
    StructType,
    NominalClassType,
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
  auto operator=(const Value&) -> Value& = delete;

  void Print(llvm::raw_ostream& out) const;
  LLVM_DUMP_METHOD void Dump() const { Print(llvm::errs()); }

  // Returns the sub-Value specified by `path`, which must be a valid field
  // path for *this.
  auto GetField(Nonnull<Arena*> arena, const FieldPath& path,
                SourceLocation source_loc) const -> Nonnull<const Value*>;

  // Returns a copy of *this, but with the sub-Value specified by `path`
  // set to `field_value`. `path` must be a valid field path for *this.
  auto SetField(Nonnull<Arena*> arena, const FieldPath& path,
                Nonnull<const Value*> field_value,
                SourceLocation source_loc) const -> Nonnull<const Value*>;

  // Returns the enumerator corresponding to the most-derived type of this
  // object.
  auto kind() const -> Kind { return kind_; }

 protected:
  // Constructs a Value. `kind` must be the enumerator corresponding to the
  // most-derived type being constructed.
  explicit Value(Kind kind) : kind_(kind) {}

 private:
  const Kind kind_;
};

using VarValues = std::vector<std::pair<std::string, Nonnull<const Value*>>>;

auto FindInVarValues(const std::string& field, const VarValues& inits)
    -> std::optional<Nonnull<const Value*>>;
auto FieldsEqual(const VarValues& ts1, const VarValues& ts2) -> bool;

// A StructElement represents the value of a single struct field.
//
// TODO(geoffromer): Look for ways to eliminate duplication among StructElement,
// VarValues::value_type, FieldInitializer, and any similar types.
struct StructElement {
  // The field name.
  std::string name;

  // The field's value.
  Nonnull<const Value*> value;
};

struct Frame;  // Used by continuation.

// An integer value.
class IntValue : public Value {
 public:
  explicit IntValue(int val) : Value(Kind::IntValue), val(val) {}

  static auto classof(const Value* value) -> bool {
    return value->kind() == Kind::IntValue;
  }

  auto Val() const -> int { return val; }

 private:
  int val;
};

// A function value.
class FunctionValue : public Value {
 public:
  FunctionValue(std::string name, Nonnull<const Value*> param,
                std::optional<Nonnull<const Statement*>> body)
      : Value(Kind::FunctionValue),
        name(std::move(name)),
        param(param),
        body(body) {}

  static auto classof(const Value* value) -> bool {
    return value->kind() == Kind::FunctionValue;
  }

  auto Name() const -> const std::string& { return name; }
  auto Param() const -> Nonnull<const Value*> { return param; }
  auto Body() const -> std::optional<Nonnull<const Statement*>> { return body; }

 private:
  std::string name;
  Nonnull<const Value*> param;
  std::optional<Nonnull<const Statement*>> body;
};

// A pointer value.
class PointerValue : public Value {
 public:
  explicit PointerValue(Address val)
      : Value(Kind::PointerValue), val(std::move(val)) {}

  static auto classof(const Value* value) -> bool {
    return value->kind() == Kind::PointerValue;
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
    return value->kind() == Kind::BoolValue;
  }

  auto Val() const -> bool { return val; }

 private:
  bool val;
};

// A non-empty value of a struct type.
//
// It can't be empty because `{}` is a struct type as well as a value of that
// type, so for consistency we always represent it as a StructType rather than
// let it oscillate unpredictably between the two. However, this means code
// that handles StructValue instances may also need to be able to handle
// StructType instances.
class StructValue : public Value {
 public:
  explicit StructValue(std::vector<StructElement> elements)
      : Value(Kind::StructValue), elements_(std::move(elements)) {
    CHECK(!elements_.empty())
        << "`{}` is represented as a StructType, not a StructValue.";
  }

  static auto classof(const Value* value) -> bool {
    return value->kind() == Kind::StructValue;
  }

  auto elements() const -> const std::vector<StructElement>& {
    return elements_;
  }

  // Returns the value of the field named `name` in this struct, or
  // nullopt if there is no such field.
  auto FindField(const std::string& name) const
      -> std::optional<Nonnull<const Value*>>;

 private:
  std::vector<StructElement> elements_;
};

// A value of a nominal class type.
class NominalClassValue : public Value {
 public:
  NominalClassValue(Nonnull<const Value*> type, Nonnull<const Value*> inits)
      : Value(Kind::NominalClassValue), type(type), inits(inits) {}

  static auto classof(const Value* value) -> bool {
    return value->kind() == Kind::NominalClassValue;
  }

  auto Type() const -> Nonnull<const Value*> { return type; }
  auto Inits() const -> Nonnull<const Value*> { return inits; }

 private:
  Nonnull<const Value*> type;
  Nonnull<const Value*> inits;
};

// An alternative constructor value.
class AlternativeConstructorValue : public Value {
 public:
  AlternativeConstructorValue(std::string alt_name, std::string choice_name)
      : Value(Kind::AlternativeConstructorValue),
        alt_name(std::move(alt_name)),
        choice_name(std::move(choice_name)) {}

  static auto classof(const Value* value) -> bool {
    return value->kind() == Kind::AlternativeConstructorValue;
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
                   Nonnull<const Value*> argument)
      : Value(Kind::AlternativeValue),
        alt_name(std::move(alt_name)),
        choice_name(std::move(choice_name)),
        argument(argument) {}

  static auto classof(const Value* value) -> bool {
    return value->kind() == Kind::AlternativeValue;
  }

  auto AltName() const -> const std::string& { return alt_name; }
  auto ChoiceName() const -> const std::string& { return choice_name; }
  auto Argument() const -> Nonnull<const Value*> { return argument; }

 private:
  std::string alt_name;
  std::string choice_name;
  Nonnull<const Value*> argument;
};

// A function value.
class TupleValue : public Value {
 public:
  // An empty tuple, also known as the unit type.
  static auto Empty() -> Nonnull<const TupleValue*> {
    static const TupleValue empty =
        TupleValue(std::vector<Nonnull<const Value*>>());
    return Nonnull<const TupleValue*>(&empty);
  }

  explicit TupleValue(std::vector<Nonnull<const Value*>> elements)
      : Value(Kind::TupleValue), elements(std::move(elements)) {}

  static auto classof(const Value* value) -> bool {
    return value->kind() == Kind::TupleValue;
  }

  auto Elements() const -> const std::vector<Nonnull<const Value*>>& {
    return elements;
  }

 private:
  std::vector<Nonnull<const Value*>> elements;
};

// A binding placeholder value.
class BindingPlaceholderValue : public Value {
 public:
  // nullopt represents the `_` placeholder.
  BindingPlaceholderValue(std::optional<std::string> name,
                          Nonnull<const Value*> type)
      : Value(Kind::BindingPlaceholderValue),
        name(std::move(name)),
        type(type) {}

  static auto classof(const Value* value) -> bool {
    return value->kind() == Kind::BindingPlaceholderValue;
  }

  auto Name() const -> const std::optional<std::string>& { return name; }
  auto Type() const -> Nonnull<const Value*> { return type; }

 private:
  std::optional<std::string> name;
  Nonnull<const Value*> type;
};

// The int type.
class IntType : public Value {
 public:
  IntType() : Value(Kind::IntType) {}

  static auto classof(const Value* value) -> bool {
    return value->kind() == Kind::IntType;
  }
};

// The bool type.
class BoolType : public Value {
 public:
  BoolType() : Value(Kind::BoolType) {}

  static auto classof(const Value* value) -> bool {
    return value->kind() == Kind::BoolType;
  }
};

// A type type.
class TypeType : public Value {
 public:
  TypeType() : Value(Kind::TypeType) {}

  static auto classof(const Value* value) -> bool {
    return value->kind() == Kind::TypeType;
  }
};

// A function type.
class FunctionType : public Value {
 public:
  FunctionType(std::vector<GenericBinding> deduced, Nonnull<const Value*> param,
               Nonnull<const Value*> ret)
      : Value(Kind::FunctionType),
        deduced(std::move(deduced)),
        param(param),
        ret(ret) {}

  static auto classof(const Value* value) -> bool {
    return value->kind() == Kind::FunctionType;
  }

  auto Deduced() const -> const std::vector<GenericBinding>& { return deduced; }
  auto Param() const -> Nonnull<const Value*> { return param; }
  auto Ret() const -> Nonnull<const Value*> { return ret; }

 private:
  std::vector<GenericBinding> deduced;
  Nonnull<const Value*> param;
  Nonnull<const Value*> ret;
};

// A pointer type.
class PointerType : public Value {
 public:
  explicit PointerType(Nonnull<const Value*> type)
      : Value(Kind::PointerType), type(type) {}

  static auto classof(const Value* value) -> bool {
    return value->kind() == Kind::PointerType;
  }

  auto Type() const -> Nonnull<const Value*> { return type; }

 private:
  Nonnull<const Value*> type;
};

// The `auto` type.
class AutoType : public Value {
 public:
  AutoType() : Value(Kind::AutoType) {}

  static auto classof(const Value* value) -> bool {
    return value->kind() == Kind::AutoType;
  }
};

// A struct type.
//
// Code that handles this type may sometimes need to have special-case handling
// for `{}`, which is a struct value in addition to being a struct type.
class StructType : public Value {
 public:
  StructType() : StructType(VarValues{}) {}

  explicit StructType(VarValues fields)
      : Value(Kind::StructType), fields_(std::move(fields)) {}

  static auto classof(const Value* value) -> bool {
    return value->kind() == Kind::StructType;
  }

  auto fields() const -> const VarValues& { return fields_; }

 private:
  VarValues fields_;
};

// A class type.
class NominalClassType : public Value {
 public:
  NominalClassType(std::string name, VarValues fields, VarValues methods)
      : Value(Kind::NominalClassType),
        name(std::move(name)),
        fields(std::move(fields)),
        methods(std::move(methods)) {}

  static auto classof(const Value* value) -> bool {
    return value->kind() == Kind::NominalClassType;
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
    return value->kind() == Kind::ChoiceType;
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
    return value->kind() == Kind::ContinuationType;
  }
};

// A variable type.
class VariableType : public Value {
 public:
  explicit VariableType(std::string name)
      : Value(Kind::VariableType), name(std::move(name)) {}

  static auto classof(const Value* value) -> bool {
    return value->kind() == Kind::VariableType;
  }

  auto Name() const -> const std::string& { return name; }

 private:
  std::string name;
};

// A first-class continuation representation of a fragment of the stack.
// A continuation value behaves like a pointer to the underlying stack
// fragment, which is exposed by `Stack()`.
class ContinuationValue : public Value {
 public:
  explicit ContinuationValue(Nonnull<std::vector<Nonnull<Frame*>>*> stack)
      : Value(Kind::ContinuationValue), stack(stack) {}

  static auto classof(const Value* value) -> bool {
    return value->kind() == Kind::ContinuationValue;
  }

  // The call stack of the suspended continuation, starting with the top
  // frame (the reverse of the usual order). Note that this provides mutable
  // access, even when *this is const, because of the reference-like semantics
  // of ContinuationValue.
  auto Stack() const -> Nonnull<std::vector<Nonnull<Frame*>>*> { return stack; }

 private:
  Nonnull<std::vector<Nonnull<Frame*>>*> stack;
};

// The String type.
class StringType : public Value {
 public:
  StringType() : Value(Kind::StringType) {}

  static auto classof(const Value* value) -> bool {
    return value->kind() == Kind::StringType;
  }
};

// A string value.
class StringValue : public Value {
 public:
  explicit StringValue(std::string val)
      : Value(Kind::StringValue), val(std::move(val)) {}

  static auto classof(const Value* value) -> bool {
    return value->kind() == Kind::StringValue;
  }

  auto Val() const -> const std::string& { return val; }

 private:
  std::string val;
};

auto TypeEqual(Nonnull<const Value*> t1, Nonnull<const Value*> t2) -> bool;
auto ValueEqual(Nonnull<const Value*> v1, Nonnull<const Value*> v2,
                SourceLocation source_loc) -> bool;

}  // namespace Carbon

#endif  // EXECUTABLE_SEMANTICS_INTERPRETER_VALUE_H_
