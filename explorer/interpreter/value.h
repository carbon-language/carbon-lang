// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXPLORER_INTERPRETER_VALUE_H_
#define EXPLORER_INTERPRETER_VALUE_H_

#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "common/ostream.h"
#include "explorer/ast/declaration.h"
#include "explorer/ast/statement.h"
#include "explorer/common/nonnull.h"
#include "explorer/interpreter/address.h"
#include "explorer/interpreter/field_path.h"
#include "explorer/interpreter/stack.h"
#include "llvm/ADT/PointerUnion.h"
#include "llvm/Support/Compiler.h"

namespace Carbon {

class Action;

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
    BoundMethodValue,
    PointerValue,
    LValue,
    BoolValue,
    StructValue,
    NominalClassValue,
    AlternativeValue,
    TupleValue,
    Witness,
    IntType,
    BoolType,
    TypeType,
    FunctionType,
    PointerType,
    AutoType,
    StructType,
    NominalClassType,
    InterfaceType,
    ChoiceType,
    ContinuationType,  // The type of a continuation.
    VariableType,      // e.g., generic type parameters.
    ParameterizedEntityName,
    MemberName,
    BindingPlaceholderValue,
    AlternativeConstructorValue,
    ContinuationValue,  // A first-class continuation value.
    StringType,
    StringValue,
    TypeOfClassType,
    TypeOfInterfaceType,
    TypeOfChoiceType,
    TypeOfParameterizedEntityName,
    TypeOfMemberName,
    StaticArrayType,
  };

  Value(const Value&) = delete;
  auto operator=(const Value&) -> Value& = delete;

  void Print(llvm::raw_ostream& out) const;
  LLVM_DUMP_METHOD void Dump() const { Print(llvm::errs()); }

  // Returns the sub-Value specified by `path`, which must be a valid field
  // path for *this.
  auto GetField(Nonnull<Arena*> arena, const FieldPath& path,
                SourceLocation source_loc) const
      -> ErrorOr<Nonnull<const Value*>>;

  // Returns a copy of *this, but with the sub-Value specified by `path`
  // set to `field_value`. `path` must be a valid field path for *this.
  auto SetField(Nonnull<Arena*> arena, const FieldPath& path,
                Nonnull<const Value*> field_value,
                SourceLocation source_loc) const
      -> ErrorOr<Nonnull<const Value*>>;

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

// A NamedValue represents a value with a name, such as a single struct field.
struct NamedValue {
  // The field name.
  std::string name;

  // The field's value.
  Nonnull<const Value*> value;
};

// An integer value.
class IntValue : public Value {
 public:
  explicit IntValue(int value) : Value(Kind::IntValue), value_(value) {}

  static auto classof(const Value* value) -> bool {
    return value->kind() == Kind::IntValue;
  }

  auto value() const -> int { return value_; }

 private:
  int value_;
};

using ImplWitnessMap =
    std::map<Nonnull<const ImplBinding*>, Nonnull<const Witness*>>;

// A function value.
class FunctionValue : public Value {
 public:
  explicit FunctionValue(Nonnull<const FunctionDeclaration*> declaration)
      : Value(Kind::FunctionValue), declaration_(declaration) {}

  explicit FunctionValue(Nonnull<const FunctionDeclaration*> declaration,
                         const BindingMap& type_args,
                         const ImplWitnessMap& wits)
      : Value(Kind::FunctionValue),
        declaration_(declaration),
        type_args_(type_args),
        witnesses_(wits) {}

  static auto classof(const Value* value) -> bool {
    return value->kind() == Kind::FunctionValue;
  }

  auto declaration() const -> const FunctionDeclaration& {
    return *declaration_;
  }

  auto type_args() const -> const BindingMap& { return type_args_; }

  auto witnesses() const
      -> const std::map<Nonnull<const ImplBinding*>, const Witness*>& {
    return witnesses_;
  }

 private:
  Nonnull<const FunctionDeclaration*> declaration_;
  BindingMap type_args_;
  ImplWitnessMap witnesses_;
};

// A bound method value. It includes the receiver object.
class BoundMethodValue : public Value {
 public:
  explicit BoundMethodValue(Nonnull<const FunctionDeclaration*> declaration,
                            Nonnull<const Value*> receiver)
      : Value(Kind::BoundMethodValue),
        declaration_(declaration),
        receiver_(receiver) {}

  explicit BoundMethodValue(Nonnull<const FunctionDeclaration*> declaration,
                            Nonnull<const Value*> receiver,
                            const BindingMap& type_args,
                            const std::map<Nonnull<const ImplBinding*>,
                                           Nonnull<const Witness*>>& wits)
      : Value(Kind::BoundMethodValue),
        declaration_(declaration),
        receiver_(receiver),
        type_args_(type_args),
        witnesses_(wits) {}

  static auto classof(const Value* value) -> bool {
    return value->kind() == Kind::BoundMethodValue;
  }

  auto declaration() const -> const FunctionDeclaration& {
    return *declaration_;
  }

  auto receiver() const -> Nonnull<const Value*> { return receiver_; }

  auto type_args() const -> const BindingMap& { return type_args_; }

  auto witnesses() const -> const ImplWitnessMap& { return witnesses_; }

 private:
  Nonnull<const FunctionDeclaration*> declaration_;
  Nonnull<const Value*> receiver_;
  BindingMap type_args_;
  ImplWitnessMap witnesses_;
};

// The value of a location in memory.
class LValue : public Value {
 public:
  explicit LValue(Address value)
      : Value(Kind::LValue), value_(std::move(value)) {}

  static auto classof(const Value* value) -> bool {
    return value->kind() == Kind::LValue;
  }

  auto address() const -> const Address& { return value_; }

 private:
  Address value_;
};

// A pointer value
class PointerValue : public Value {
 public:
  explicit PointerValue(Address value)
      : Value(Kind::PointerValue), value_(std::move(value)) {}

  static auto classof(const Value* value) -> bool {
    return value->kind() == Kind::PointerValue;
  }

  auto address() const -> const Address& { return value_; }

 private:
  Address value_;
};

// A bool value.
class BoolValue : public Value {
 public:
  explicit BoolValue(bool value) : Value(Kind::BoolValue), value_(value) {}

  static auto classof(const Value* value) -> bool {
    return value->kind() == Kind::BoolValue;
  }

  auto value() const -> bool { return value_; }

 private:
  bool value_;
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
  explicit StructValue(std::vector<NamedValue> elements)
      : Value(Kind::StructValue), elements_(std::move(elements)) {
    CARBON_CHECK(!elements_.empty())
        << "`{}` is represented as a StructType, not a StructValue.";
  }

  static auto classof(const Value* value) -> bool {
    return value->kind() == Kind::StructValue;
  }

  auto elements() const -> llvm::ArrayRef<NamedValue> { return elements_; }

  // Returns the value of the field named `name` in this struct, or
  // nullopt if there is no such field.
  auto FindField(const std::string& name) const
      -> std::optional<Nonnull<const Value*>>;

 private:
  std::vector<NamedValue> elements_;
};

// A value of a nominal class type, i.e., an object.
class NominalClassValue : public Value {
 public:
  NominalClassValue(Nonnull<const Value*> type, Nonnull<const Value*> inits)
      : Value(Kind::NominalClassValue), type_(type), inits_(inits) {}

  static auto classof(const Value* value) -> bool {
    return value->kind() == Kind::NominalClassValue;
  }

  auto type() const -> const Value& { return *type_; }
  auto inits() const -> const Value& { return *inits_; }

 private:
  Nonnull<const Value*> type_;
  Nonnull<const Value*> inits_;  // The initializing StructValue.
};

// An alternative constructor value.
class AlternativeConstructorValue : public Value {
 public:
  AlternativeConstructorValue(std::string alt_name, std::string choice_name)
      : Value(Kind::AlternativeConstructorValue),
        alt_name_(std::move(alt_name)),
        choice_name_(std::move(choice_name)) {}

  static auto classof(const Value* value) -> bool {
    return value->kind() == Kind::AlternativeConstructorValue;
  }

  auto alt_name() const -> const std::string& { return alt_name_; }
  auto choice_name() const -> const std::string& { return choice_name_; }

 private:
  std::string alt_name_;
  std::string choice_name_;
};

// An alternative value.
class AlternativeValue : public Value {
 public:
  AlternativeValue(std::string alt_name, std::string choice_name,
                   Nonnull<const Value*> argument)
      : Value(Kind::AlternativeValue),
        alt_name_(std::move(alt_name)),
        choice_name_(std::move(choice_name)),
        argument_(argument) {}

  static auto classof(const Value* value) -> bool {
    return value->kind() == Kind::AlternativeValue;
  }

  auto alt_name() const -> const std::string& { return alt_name_; }
  auto choice_name() const -> const std::string& { return choice_name_; }
  auto argument() const -> const Value& { return *argument_; }

 private:
  std::string alt_name_;
  std::string choice_name_;
  Nonnull<const Value*> argument_;
};

// A tuple value.
class TupleValue : public Value {
 public:
  // An empty tuple, also known as the unit type.
  static auto Empty() -> Nonnull<const TupleValue*> {
    static const TupleValue empty =
        TupleValue(std::vector<Nonnull<const Value*>>());
    return Nonnull<const TupleValue*>(&empty);
  }

  explicit TupleValue(std::vector<Nonnull<const Value*>> elements)
      : Value(Kind::TupleValue), elements_(std::move(elements)) {}

  static auto classof(const Value* value) -> bool {
    return value->kind() == Kind::TupleValue;
  }

  auto elements() const -> llvm::ArrayRef<Nonnull<const Value*>> {
    return elements_;
  }

 private:
  std::vector<Nonnull<const Value*>> elements_;
};

// A binding placeholder value.
class BindingPlaceholderValue : public Value {
 public:
  // Represents the `_` placeholder.
  explicit BindingPlaceholderValue() : Value(Kind::BindingPlaceholderValue) {}

  // Represents a named placeholder.
  explicit BindingPlaceholderValue(ValueNodeView value_node)
      : Value(Kind::BindingPlaceholderValue),
        value_node_(std::move(value_node)) {}

  static auto classof(const Value* value) -> bool {
    return value->kind() == Kind::BindingPlaceholderValue;
  }

  auto value_node() const -> const std::optional<ValueNodeView>& {
    return value_node_;
  }

 private:
  std::optional<ValueNodeView> value_node_;
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
  FunctionType(llvm::ArrayRef<Nonnull<const GenericBinding*>> deduced,
               Nonnull<const Value*> parameters,
               Nonnull<const Value*> return_type,
               llvm::ArrayRef<Nonnull<const ImplBinding*>> impl_bindings)
      : Value(Kind::FunctionType),
        deduced_(deduced),
        parameters_(parameters),
        return_type_(return_type),
        impl_bindings_(impl_bindings) {}

  static auto classof(const Value* value) -> bool {
    return value->kind() == Kind::FunctionType;
  }

  auto deduced() const -> llvm::ArrayRef<Nonnull<const GenericBinding*>> {
    return deduced_;
  }
  auto parameters() const -> const Value& { return *parameters_; }
  auto return_type() const -> const Value& { return *return_type_; }
  // The bindings for the witness tables (impls) required by the
  // bounds on the type parameters of the generic function.
  auto impl_bindings() const -> llvm::ArrayRef<Nonnull<const ImplBinding*>> {
    return impl_bindings_;
  }

 private:
  std::vector<Nonnull<const GenericBinding*>> deduced_;
  Nonnull<const Value*> parameters_;
  Nonnull<const Value*> return_type_;
  std::vector<Nonnull<const ImplBinding*>> impl_bindings_;
};

// A pointer type.
class PointerType : public Value {
 public:
  explicit PointerType(Nonnull<const Value*> type)
      : Value(Kind::PointerType), type_(type) {}

  static auto classof(const Value* value) -> bool {
    return value->kind() == Kind::PointerType;
  }

  auto type() const -> const Value& { return *type_; }

 private:
  Nonnull<const Value*> type_;
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
  StructType() : StructType(std::vector<NamedValue>{}) {}

  explicit StructType(std::vector<NamedValue> fields)
      : Value(Kind::StructType), fields_(std::move(fields)) {}

  static auto classof(const Value* value) -> bool {
    return value->kind() == Kind::StructType;
  }

  auto fields() const -> llvm::ArrayRef<NamedValue> { return fields_; }

 private:
  std::vector<NamedValue> fields_;
};

// A class type.
// TODO: Consider splitting this class into several classes.
class NominalClassType : public Value {
 public:
  // Construct a non-generic class type.
  explicit NominalClassType(Nonnull<const ClassDeclaration*> declaration)
      : Value(Kind::NominalClassType), declaration_(declaration) {
    CARBON_CHECK(!declaration->type_params().has_value())
        << "missing arguments for parameterized class type";
  }

  // Construct a class type that represents the result of applying the
  // given generic class to the `type_args`.
  explicit NominalClassType(Nonnull<const ClassDeclaration*> declaration,
                            const BindingMap& type_args)
      : Value(Kind::NominalClassType),
        declaration_(declaration),
        type_args_(type_args) {}

  // Construct a class type that represents the result of applying the
  // given generic class to the `type_args` and that records the result of the
  // compile-time search for any required impls.
  explicit NominalClassType(Nonnull<const ClassDeclaration*> declaration,
                            const BindingMap& type_args,
                            const ImplExpMap& impls)
      : Value(Kind::NominalClassType),
        declaration_(declaration),
        type_args_(type_args),
        impls_(impls) {}

  // Construct a fully instantiated generic class type to represent the
  // run-time type of an object.
  explicit NominalClassType(Nonnull<const ClassDeclaration*> declaration,
                            const BindingMap& type_args,
                            const std::map<Nonnull<const ImplBinding*>,
                                           Nonnull<const Witness*>>& wits)
      : Value(Kind::NominalClassType),
        declaration_(declaration),
        type_args_(type_args),
        witnesses_(wits) {}

  static auto classof(const Value* value) -> bool {
    return value->kind() == Kind::NominalClassType;
  }

  auto declaration() const -> const ClassDeclaration& { return *declaration_; }
  auto type_args() const -> const BindingMap& { return type_args_; }

  // Maps each of an instantiated generic class's impl bindings to an
  // expression that constructs the witness table for the corresponding
  // argument. Should not be called on 1) a non-generic class, 2) a
  // generic-class that is not instantiated, or 3) a fully
  // instantiated runtime type of a generic class.
  auto impls() const -> const ImplExpMap& { return impls_; }

  // Maps each of the class's impl bindings to the witness table
  // for the corresponding argument. Should only be called on a fully
  // instantiated runtime type of a generic class.
  auto witnesses() const -> const ImplWitnessMap& { return witnesses_; }

  // Returns whether this a parameterized class. That is, a class with
  // parameters and no corresponding arguments.
  auto IsParameterized() const -> bool {
    return declaration_->type_params().has_value() && type_args_.empty();
  }

  // Returns the value of the function named `name` in this class, or
  // nullopt if there is no such function.
  auto FindFunction(const std::string& name) const
      -> std::optional<Nonnull<const FunctionValue*>>;

 private:
  Nonnull<const ClassDeclaration*> declaration_;
  BindingMap type_args_;
  ImplExpMap impls_;
  ImplWitnessMap witnesses_;
};

// Return the declaration of the member with the given name.
auto FindMember(const std::string& name,
                llvm::ArrayRef<Nonnull<Declaration*>> members)
    -> std::optional<Nonnull<const Declaration*>>;

// An interface type.
class InterfaceType : public Value {
 public:
  explicit InterfaceType(Nonnull<const InterfaceDeclaration*> declaration)
      : Value(Kind::InterfaceType), declaration_(declaration) {
    CARBON_CHECK(!declaration->params().has_value())
        << "missing arguments for parameterized interface type";
  }
  explicit InterfaceType(Nonnull<const InterfaceDeclaration*> declaration,
                         const BindingMap& args)
      : Value(Kind::InterfaceType), declaration_(declaration), args_(args) {}
  explicit InterfaceType(Nonnull<const InterfaceDeclaration*> declaration,
                         const BindingMap& args, const ImplExpMap& impls)
      : Value(Kind::InterfaceType),
        declaration_(declaration),
        args_(args),
        impls_(impls) {}
  explicit InterfaceType(Nonnull<const InterfaceDeclaration*> declaration,
                         const BindingMap& args, const ImplWitnessMap& wits)
      : Value(Kind::InterfaceType),
        declaration_(declaration),
        args_(args),
        witnesses_(wits) {}

  static auto classof(const Value* value) -> bool {
    return value->kind() == Kind::InterfaceType;
  }

  auto declaration() const -> const InterfaceDeclaration& {
    return *declaration_;
  }
  auto args() const -> const BindingMap& { return args_; }

  // FIXME: These aren't used for anything yet.
  auto impls() const -> const ImplExpMap& { return impls_; }
  auto witnesses() const -> const ImplWitnessMap& { return witnesses_; }

 private:
  Nonnull<const InterfaceDeclaration*> declaration_;
  BindingMap args_;
  ImplExpMap impls_;
  ImplWitnessMap witnesses_;
};

// The witness table for an impl.
class Witness : public Value {
 public:
  // Construct a witness for
  // 1) a non-generic impl, or
  // 2) a generic impl that has not yet been applied to type arguments.
  explicit Witness(Nonnull<const ImplDeclaration*> declaration)
      : Value(Kind::Witness), declaration_(declaration) {}

  // Construct an instantiated generic impl.
  explicit Witness(Nonnull<const ImplDeclaration*> declaration,
                   const BindingMap& type_args, const ImplWitnessMap& wits)
      : Value(Kind::Witness),
        declaration_(declaration),
        type_args_(type_args),
        witnesses_(wits) {}

  static auto classof(const Value* value) -> bool {
    return value->kind() == Kind::Witness;
  }
  auto declaration() const -> const ImplDeclaration& { return *declaration_; }
  auto type_args() const -> const BindingMap& { return type_args_; }
  // Maps each of the impl's impl bindings to the witness table
  // for the corresponding argument. Should only be called on a fully
  // instantiated runtime type of a generic class.
  auto witnesses() const -> const ImplWitnessMap& { return witnesses_; }

 private:
  Nonnull<const ImplDeclaration*> declaration_;
  BindingMap type_args_;
  ImplWitnessMap witnesses_;
};

// A choice type.
class ChoiceType : public Value {
 public:
  ChoiceType(std::string name, std::vector<NamedValue> alternatives)
      : Value(Kind::ChoiceType),
        name_(std::move(name)),
        alternatives_(std::move(alternatives)) {}

  static auto classof(const Value* value) -> bool {
    return value->kind() == Kind::ChoiceType;
  }

  auto name() const -> const std::string& { return name_; }

  // Returns the parameter types of the alternative with the given name,
  // or nullopt if no such alternative is present.
  auto FindAlternative(std::string_view name) const
      -> std::optional<Nonnull<const Value*>>;

 private:
  std::string name_;
  std::vector<NamedValue> alternatives_;
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
  explicit VariableType(Nonnull<const GenericBinding*> binding)
      : Value(Kind::VariableType), binding_(binding) {}

  static auto classof(const Value* value) -> bool {
    return value->kind() == Kind::VariableType;
  }

  auto binding() const -> const GenericBinding& { return *binding_; }

 private:
  Nonnull<const GenericBinding*> binding_;
};

// A name of an entity that has explicit parameters, such as a parameterized
// class or interface. When arguments for those parameters are provided in a
// call, the result will be a class type or interface type.
class ParameterizedEntityName : public Value {
 public:
  explicit ParameterizedEntityName(Nonnull<const Declaration*> declaration,
                                   Nonnull<const TuplePattern*> params)
      : Value(Kind::ParameterizedEntityName),
        declaration_(declaration),
        params_(params) {}

  static auto classof(const Value* value) -> bool {
    return value->kind() == Kind::ParameterizedEntityName;
  }

  auto declaration() const -> const Declaration& { return *declaration_; }
  auto params() const -> const TuplePattern& { return *params_; }

 private:
  Nonnull<const Declaration*> declaration_;
  Nonnull<const TuplePattern*> params_;
};

// A member of a type.
//
// This is either a declared member of a class, interface, or similar, or a
// member of a struct with no declaration.
class Member {
 public:
  explicit Member(const Declaration* declaration) : member_(declaration) {}
  explicit Member(const NamedValue* struct_member) : member_(struct_member) {}

  // The name of the member.
  auto name() const -> std::string;
  // The declared type of the member, which might include type variables.
  auto type() const -> const Value&;
  // A declaration of the member, if any exists.
  auto declaration() const -> std::optional<Nonnull<const Declaration*>> {
    if (const Declaration* decl = member_.dyn_cast<const Declaration*>()) {
      return decl;
    }
    return std::nullopt;
  }

 private:
  llvm::PointerUnion<Nonnull<const Declaration*>, Nonnull<const NamedValue*>>
      member_;
};

// The name of a member of a class or interface.
//
// These values are used to represent the second operand of a compound member
// access expression: `x.(A.B)`.
class MemberName : public Value {
 public:
  MemberName(std::optional<Nonnull<const Value*>> base_type,
             std::optional<Nonnull<const InterfaceType*>> interface,
             Member member)
      : Value(Kind::MemberName),
        base_type_(base_type),
        interface_(interface),
        member_(member) {
    CARBON_CHECK(base_type || interface)
        << "member name must be in a type, an interface, or both";
  }

  static auto classof(const Value* value) -> bool {
    return value->kind() == Kind::MemberName;
  }

  // The type for which `name` is a member or a member of an `impl`.
  auto base_type() const -> std::optional<Nonnull<const Value*>> {
    return base_type_;
  }
  // The interface for which `name` is a member, if any.
  auto interface() const -> std::optional<Nonnull<const InterfaceType*>> {
    return interface_;
  }
  // The member.
  auto member() const -> Member { return member_; }
  // The name of the member.
  auto name() const -> std::string { return member().name(); }

 private:
  std::optional<Nonnull<const Value*>> base_type_;
  std::optional<Nonnull<const InterfaceType*>> interface_;
  Member member_;
};

// A first-class continuation representation of a fragment of the stack.
// A continuation value behaves like a pointer to the underlying stack
// fragment, which is exposed by `Stack()`.
class ContinuationValue : public Value {
 public:
  class StackFragment {
   public:
    // Constructs an empty StackFragment.
    StackFragment() = default;

    // Requires *this to be empty, because by the time we're tearing down the
    // Arena, it's no longer safe to invoke ~Action.
    ~StackFragment();

    StackFragment(StackFragment&&) = delete;
    auto operator=(StackFragment&&) -> StackFragment& = delete;

    // Store the given partial todo stack in *this, which must currently be
    // empty. The stack is represented with the top of the stack at the
    // beginning of the vector, the reverse of the usual order.
    void StoreReversed(std::vector<std::unique_ptr<Action>> reversed_todo);

    // Restore the currently stored stack fragment to the top of `todo`,
    // leaving *this empty.
    void RestoreTo(Stack<std::unique_ptr<Action>>& todo);

    // Destroy the currently stored stack fragment.
    void Clear();

    void Print(llvm::raw_ostream& out) const;
    LLVM_DUMP_METHOD void Dump() const { Print(llvm::errs()); }

   private:
    // The todo stack of a suspended continuation, starting with the top
    // Action.
    std::vector<std::unique_ptr<Action>> reversed_todo_;
  };

  explicit ContinuationValue(Nonnull<StackFragment*> stack)
      : Value(Kind::ContinuationValue), stack_(stack) {}

  static auto classof(const Value* value) -> bool {
    return value->kind() == Kind::ContinuationValue;
  }

  // The todo stack of the suspended continuation. Note that this provides
  // mutable access, even when *this is const, because of the reference-like
  // semantics of ContinuationValue.
  auto stack() const -> StackFragment& { return *stack_; }

 private:
  Nonnull<StackFragment*> stack_;
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
  explicit StringValue(std::string value)
      : Value(Kind::StringValue), value_(std::move(value)) {}

  static auto classof(const Value* value) -> bool {
    return value->kind() == Kind::StringValue;
  }

  auto value() const -> const std::string& { return value_; }

 private:
  std::string value_;
};

// The type of an expression whose value is a class type. Currently there is no
// way to explicitly name such a type in Carbon code, but we are tentatively
// using `typeof(ClassName)` as the debug-printing format, in anticipation of
// something like that becoming valid Carbon syntax.
class TypeOfClassType : public Value {
 public:
  explicit TypeOfClassType(Nonnull<const NominalClassType*> class_type)
      : Value(Kind::TypeOfClassType), class_type_(class_type) {}

  static auto classof(const Value* value) -> bool {
    return value->kind() == Kind::TypeOfClassType;
  }

  auto class_type() const -> const NominalClassType& { return *class_type_; }

 private:
  Nonnull<const NominalClassType*> class_type_;
};

class TypeOfInterfaceType : public Value {
 public:
  explicit TypeOfInterfaceType(Nonnull<const InterfaceType*> iface_type)
      : Value(Kind::TypeOfInterfaceType), iface_type_(iface_type) {}

  static auto classof(const Value* value) -> bool {
    return value->kind() == Kind::TypeOfInterfaceType;
  }

  auto interface_type() const -> const InterfaceType& { return *iface_type_; }

 private:
  Nonnull<const InterfaceType*> iface_type_;
};

// The type of an expression whose value is a choice type. Currently there is no
// way to explicitly name such a type in Carbon code, but we are tentatively
// using `typeof(ChoiceName)` as the debug-printing format, in anticipation of
// something like that becoming valid Carbon syntax.
class TypeOfChoiceType : public Value {
 public:
  explicit TypeOfChoiceType(Nonnull<const ChoiceType*> choice_type)
      : Value(Kind::TypeOfChoiceType), choice_type_(choice_type) {}

  static auto classof(const Value* value) -> bool {
    return value->kind() == Kind::TypeOfChoiceType;
  }

  auto choice_type() const -> const ChoiceType& { return *choice_type_; }

 private:
  Nonnull<const ChoiceType*> choice_type_;
};

// The type of an expression whose value is the name of a parameterized entity.
// Such an expression can only be used as the operand of a call expression that
// provides arguments for the parameters.
class TypeOfParameterizedEntityName : public Value {
 public:
  explicit TypeOfParameterizedEntityName(
      Nonnull<const ParameterizedEntityName*> name)
      : Value(Kind::TypeOfParameterizedEntityName), name_(name) {}

  static auto classof(const Value* value) -> bool {
    return value->kind() == Kind::TypeOfParameterizedEntityName;
  }

  auto name() const -> const ParameterizedEntityName& { return *name_; }

 private:
  Nonnull<const ParameterizedEntityName*> name_;
};

// The type of a member name expression.
//
// This is used for member names that don't denote a specific object or value
// until used on the right-hand side of a `.`, such as an instance method or
// field name, or any member function in an interface.
//
// Such expressions can appear only as the target of an `alias` declaration or
// as the member name in a compound member access.
class TypeOfMemberName : public Value {
 public:
  explicit TypeOfMemberName(Member member)
      : Value(Kind::TypeOfMemberName), member_(member) {}

  static auto classof(const Value* value) -> bool {
    return value->kind() == Kind::TypeOfMemberName;
  }

  auto member() const -> Member { return member_; }

 private:
  Member member_;
};

// The type of a statically-sized array.
//
// Note that values of this type are represented as tuples.
class StaticArrayType : public Value {
 public:
  // Constructs a statically-sized array type with the given element type and
  // size.
  StaticArrayType(Nonnull<const Value*> element_type, size_t size)
      : Value(Kind::StaticArrayType),
        element_type_(element_type),
        size_(size) {}

  static auto classof(const Value* value) -> bool {
    return value->kind() == Kind::StaticArrayType;
  }

  auto element_type() const -> const Value& { return *element_type_; }
  auto size() const -> size_t { return size_; }

 private:
  Nonnull<const Value*> element_type_;
  size_t size_;
};

auto TypeEqual(Nonnull<const Value*> t1, Nonnull<const Value*> t2) -> bool;
auto ValueEqual(Nonnull<const Value*> v1, Nonnull<const Value*> v2) -> bool;

}  // namespace Carbon

#endif  // EXPLORER_INTERPRETER_VALUE_H_
