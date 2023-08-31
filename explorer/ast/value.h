// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_EXPLORER_AST_VALUE_H_
#define CARBON_EXPLORER_AST_VALUE_H_

#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "common/ostream.h"
#include "explorer/ast/address.h"
#include "explorer/ast/bindings.h"
#include "explorer/ast/declaration.h"
#include "explorer/ast/element.h"
#include "explorer/ast/element_path.h"
#include "explorer/ast/expression_category.h"
#include "explorer/ast/statement.h"
#include "explorer/base/nonnull.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Compiler.h"

namespace Carbon {

class AssociatedConstant;
class ChoiceType;
class TupleValue;

// A trait type that describes how to allocate an instance of `T` in an arena.
// Returns the created object, which is not required to be of type `T`.
template <typename T>
struct AllocateTrait {
  template <typename... Args>
  static auto New(Nonnull<Arena*> arena, Args&&... args) -> Nonnull<const T*> {
    return arena->New<T>(std::forward<Args>(args)...);
  }
};

using VTable =
    llvm::StringMap<std::pair<Nonnull<const CallableDeclaration*>, int>>;

// Returns a pointer to an empty VTable that will never be deallocated.
//
// Using this instead of `new VTable()` avoids unnecessary allocations, and
// takes better advantage of Arena canonicalization when a VTable pointer is
// used as a constructor argument.
inline auto EmptyVTable() -> Nonnull<const VTable*> {
  static Nonnull<const VTable*> result = new VTable();
  return result;
}

// Abstract base class of all AST nodes representing values.
//
// Value and its derived classes support LLVM-style RTTI, including
// llvm::isa, llvm::cast, and llvm::dyn_cast. To support this, every
// class derived from Value must provide a `classof` operation, and
// every concrete derived class must have a corresponding enumerator
// in `Kind`; see https://llvm.org/docs/HowToSetUpLLVMStyleRTTI.html for
// details.
//
// Arena's canonicalization support is enabled for Value and all derived types.
// As a result, all Values must be immutable, and all their constructor
// arguments must be copyable, equality-comparable, and hashable. See
// Arena's documentation for details.
class Value : public Printable<Value> {
 public:
  using EnableCanonicalizedAllocation = void;
  enum class Kind {
#define CARBON_VALUE_KIND(kind) kind,
#include "explorer/ast/value_kinds.def"
  };

  Value(const Value&) = delete;
  auto operator=(const Value&) -> Value& = delete;

  // Call `f` on this value, cast to its most-derived type. `R` specifies the
  // expected return type of `f`.
  template <typename R, typename F>
  auto Visit(F f) const -> R;

  void Print(llvm::raw_ostream& out) const;

  // Returns the sub-Value specified by `path`, which must be a valid element
  // path for *this. If the sub-Value is a method and its self_pattern is an
  // AddrPattern, then pass the LocationValue representing the receiver as
  // `me_value`, otherwise pass `*this`.
  auto GetElement(Nonnull<Arena*> arena, const ElementPath& path,
                  SourceLocation source_loc,
                  std::optional<Nonnull<const Value*>> me_value) const
      -> ErrorOr<Nonnull<const Value*>>;

  // Returns a copy of *this, but with the sub-Value specified by `path`
  // set to `field_value`. `path` must be a valid field path for *this.
  auto SetField(Nonnull<Arena*> arena, const ElementPath& path,
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

// Returns whether the fully-resolved kind that this value will eventually have
// is currently unknown, because it depends on a generic parameter.
inline auto IsValueKindDependent(Nonnull<const Value*> type) -> bool {
  return type->kind() == Value::Kind::VariableType ||
         type->kind() == Value::Kind::AssociatedConstant;
}

// Base class for types holding contextual information by which we can
// determine whether values are equal.
class EqualityContext {
 public:
  virtual auto VisitEqualValues(
      Nonnull<const Value*> value,
      llvm::function_ref<bool(Nonnull<const Value*>)> visitor) const
      -> bool = 0;

 protected:
  virtual ~EqualityContext() = default;
};

auto TypeEqual(Nonnull<const Value*> t1, Nonnull<const Value*> t2,
               std::optional<Nonnull<const EqualityContext*>> equality_ctx)
    -> bool;
auto ValueEqual(Nonnull<const Value*> v1, Nonnull<const Value*> v2,
                std::optional<Nonnull<const EqualityContext*>> equality_ctx)
    -> bool;

// Call the given `visitor` on all values nested within the given value,
// including `value` itself, in a preorder traversal. Aborts and returns
// `false` if `visitor` returns `false`, otherwise returns `true`.
auto VisitNestedValues(Nonnull<const Value*> value,
                       llvm::function_ref<bool(const Value*)> visitor) -> bool;

// An integer value.
class IntValue : public Value {
 public:
  explicit IntValue(int value) : Value(Kind::IntValue), value_(value) {}

  static auto classof(const Value* value) -> bool {
    return value->kind() == Kind::IntValue;
  }

  template <typename F>
  auto Decompose(F f) const {
    return f(value_);
  }

  auto value() const -> int { return value_; }

 private:
  int value_;
};

// A function or bound method value.
class FunctionOrMethodValue : public Value {
 public:
  explicit FunctionOrMethodValue(
      Kind kind, Nonnull<const FunctionDeclaration*> declaration,
      Nonnull<const Bindings*> bindings)
      : Value(kind), declaration_(declaration), bindings_(bindings) {}

  static auto classof(const Value* value) -> bool {
    return value->kind() == Kind::FunctionValue ||
           value->kind() == Kind::BoundMethodValue;
  }

  auto declaration() const -> const FunctionDeclaration& {
    return *declaration_;
  }

  auto bindings() const -> const Bindings& { return *bindings_; }

  auto type_args() const -> const BindingMap& { return bindings_->args(); }

  auto witnesses() const -> const ImplWitnessMap& {
    return bindings_->witnesses();
  }

 private:
  Nonnull<const FunctionDeclaration*> declaration_;
  Nonnull<const Bindings*> bindings_;
};

// A function value.
class FunctionValue : public FunctionOrMethodValue {
 public:
  explicit FunctionValue(Nonnull<const FunctionDeclaration*> declaration,
                         Nonnull<const Bindings*> bindings)
      : FunctionOrMethodValue(Kind::FunctionValue, declaration, bindings) {}

  static auto classof(const Value* value) -> bool {
    return value->kind() == Kind::FunctionValue;
  }

  template <typename F>
  auto Decompose(F f) const {
    return f(&declaration(), &bindings());
  }
};

// A bound method value. It includes the receiver object.
class BoundMethodValue : public FunctionOrMethodValue {
 public:
  explicit BoundMethodValue(Nonnull<const FunctionDeclaration*> declaration,
                            Nonnull<const Value*> receiver,
                            Nonnull<const Bindings*> bindings)
      : FunctionOrMethodValue(Kind::BoundMethodValue, declaration, bindings),
        receiver_(receiver) {}

  static auto classof(const Value* value) -> bool {
    return value->kind() == Kind::BoundMethodValue;
  }

  template <typename F>
  auto Decompose(F f) const {
    return f(&declaration(), receiver_, &bindings());
  }

  auto receiver() const -> Nonnull<const Value*> { return receiver_; }

 private:
  Nonnull<const Value*> receiver_;
};

// A destructor value.
class DestructorValue : public Value {
 public:
  explicit DestructorValue(Nonnull<const DestructorDeclaration*> declaration)
      : Value(Kind::DestructorValue), declaration_(declaration) {}

  static auto classof(const Value* value) -> bool {
    return value->kind() == Kind::DestructorValue;
  }

  template <typename F>
  auto Decompose(F f) const {
    return f(declaration_);
  }

  auto declaration() const -> const DestructorDeclaration& {
    return *declaration_;
  }

 private:
  Nonnull<const DestructorDeclaration*> declaration_;
};

// The value of a location in memory.
class LocationValue : public Value {
 public:
  explicit LocationValue(Address value)
      : Value(Kind::LocationValue), value_(std::move(value)) {}

  static auto classof(const Value* value) -> bool {
    return value->kind() == Kind::LocationValue;
  }

  template <typename F>
  auto Decompose(F f) const {
    return f(value_);
  }

  auto address() const -> const Address& { return value_; }

 private:
  Address value_;
};

// Contains the result of the evaluation of an expression, including a value,
// the original expression category, and an optional address if available.
class ExpressionResult {
 public:
  static auto Value(Nonnull<const Carbon::Value*> v) -> ExpressionResult {
    return ExpressionResult(v, std::nullopt, ExpressionCategory::Value);
  }
  static auto Reference(Nonnull<const Carbon::Value*> v, Address address)
      -> ExpressionResult {
    return ExpressionResult(v, std::move(address),
                            ExpressionCategory::Reference);
  }
  static auto Initializing(Nonnull<const Carbon::Value*> v, Address address)
      -> ExpressionResult {
    return ExpressionResult(v, std::move(address),
                            ExpressionCategory::Initializing);
  }

  ExpressionResult(Nonnull<const Carbon::Value*> v,
                   std::optional<Address> address, ExpressionCategory cat)
      : value_(v), address_(std::move(address)), expr_cat_(cat) {}

  auto value() const -> Nonnull<const Carbon::Value*> { return value_; }
  auto address() const -> const std::optional<Address>& { return address_; }
  auto expression_category() const -> ExpressionCategory { return expr_cat_; }

 private:
  Nonnull<const Carbon::Value*> value_;
  std::optional<Address> address_;
  ExpressionCategory expr_cat_;
};

// Represents the result of the evaluation of a reference expression, and
// holds the resulting `Value*` and its `Address`.
class ReferenceExpressionValue : public Value {
 public:
  ReferenceExpressionValue(Nonnull<const Value*> value, Address address)
      : Value(Kind::ReferenceExpressionValue),
        value_(value),
        address_(std::move(address)) {}

  static auto classof(const Value* value) -> bool {
    return value->kind() == Kind::ReferenceExpressionValue;
  }

  template <typename F>
  auto Decompose(F f) const {
    return f(value_, address_);
  }

  auto value() const -> Nonnull<const Value*> { return value_; }
  auto address() const -> const Address& { return address_; }

 private:
  Nonnull<const Value*> value_;
  Address address_;
};

// A pointer value
class PointerValue : public Value {
 public:
  explicit PointerValue(Address value)
      : Value(Kind::PointerValue), value_(std::move(value)) {}

  static auto classof(const Value* value) -> bool {
    return value->kind() == Kind::PointerValue;
  }

  template <typename F>
  auto Decompose(F f) const {
    return f(value_);
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

  template <typename F>
  auto Decompose(F f) const {
    return f(value_);
  }

  auto value() const -> bool { return value_; }

 private:
  bool value_;
};

// A value of a struct type. Note that the expression `{}` is a value of type
// `{} as type`; the former is a `StructValue` and the latter is a
// `StructType`.
class StructValue : public Value {
 public:
  explicit StructValue(std::vector<NamedValue> elements)
      : Value(Kind::StructValue), elements_(std::move(elements)) {}

  static auto classof(const Value* value) -> bool {
    return value->kind() == Kind::StructValue;
  }

  template <typename F>
  auto Decompose(F f) const {
    return f(elements_);
  }

  auto elements() const -> llvm::ArrayRef<NamedValue> { return elements_; }

  // Returns the value of the field named `name` in this struct, or
  // nullopt if there is no such field.
  auto FindField(std::string_view name) const
      -> std::optional<Nonnull<const Value*>>;

 private:
  std::vector<NamedValue> elements_;
};

// A value of a nominal class type, i.e., an object.
class NominalClassValue : public Value {
 public:
  static constexpr llvm::StringLiteral BaseField{"base"};

  // Takes the class type, inits, an optional base, a pointer to a
  // NominalClassValue*, that must be common to all NominalClassValue of the
  // same object. The pointee is updated, when `NominalClassValue`s are
  // constructed, to point to the `NominalClassValue` corresponding to the
  // child-most class type. Sets *class_value_ptr = this, which corresponds to
  // the static type of the value matching its dynamic type.
  NominalClassValue(Nonnull<const Value*> type, Nonnull<const Value*> inits,
                    std::optional<Nonnull<const NominalClassValue*>> base,
                    Nonnull<const NominalClassValue** const> class_value_ptr);

  static auto classof(const Value* value) -> bool {
    return value->kind() == Kind::NominalClassValue;
  }

  template <typename F>
  auto Decompose(F f) const {
    return f(type_, inits_, base_, class_value_ptr_);
  }

  auto type() const -> const Value& { return *type_; }
  auto inits() const -> const Value& { return *inits_; }
  auto base() const -> std::optional<Nonnull<const NominalClassValue*>> {
    return base_;
  }
  // Returns a pointer of pointer to the child-most class value.
  auto class_value_ptr() const -> Nonnull<const NominalClassValue**> {
    return class_value_ptr_;
  }

 private:
  Nonnull<const Value*> type_;
  Nonnull<const Value*> inits_;  // The initializing StructValue.
  std::optional<Nonnull<const NominalClassValue*>> base_;
  Nonnull<const NominalClassValue** const> class_value_ptr_;
};

// An alternative constructor value.
class AlternativeConstructorValue : public Value {
 public:
  AlternativeConstructorValue(Nonnull<const ChoiceType*> choice,
                              Nonnull<const AlternativeSignature*> alternative)
      : Value(Kind::AlternativeConstructorValue),
        choice_(choice),
        alternative_(alternative) {}

  static auto classof(const Value* value) -> bool {
    return value->kind() == Kind::AlternativeConstructorValue;
  }

  template <typename F>
  auto Decompose(F f) const {
    return f(&choice(), &alternative());
  }

  auto choice() const -> const ChoiceType& { return *choice_; }
  auto alternative() const -> const AlternativeSignature& {
    return *alternative_;
  }

 private:
  Nonnull<const ChoiceType*> choice_;
  Nonnull<const AlternativeSignature*> alternative_;
};

// An alternative value.
class AlternativeValue : public Value {
 public:
  AlternativeValue(Nonnull<const ChoiceType*> choice,
                   Nonnull<const AlternativeSignature*> alternative,
                   std::optional<Nonnull<const TupleValue*>> argument)
      : Value(Kind::AlternativeValue),
        choice_(choice),
        alternative_(alternative),
        argument_(argument) {}

  static auto classof(const Value* value) -> bool {
    return value->kind() == Kind::AlternativeValue;
  }

  template <typename F>
  auto Decompose(F f) const {
    return f(&choice(), &alternative(), argument_);
  }

  auto choice() const -> const ChoiceType& { return *choice_; }
  auto alternative() const -> const AlternativeSignature& {
    return *alternative_;
  }
  auto argument() const -> std::optional<Nonnull<const TupleValue*>> {
    return argument_;
  }

 private:
  Nonnull<const ChoiceType*> choice_;
  Nonnull<const AlternativeSignature*> alternative_;
  std::optional<Nonnull<const TupleValue*>> argument_;
};

// Base class for tuple types and tuple values. These are the same other than
// their type-of-type, but we separate them to make it easier to tell types and
// values apart.
class TupleValueBase : public Value {
 public:
  explicit TupleValueBase(Value::Kind kind,
                          std::vector<Nonnull<const Value*>> elements)
      : Value(kind), elements_(std::move(elements)) {}

  auto elements() const -> llvm::ArrayRef<Nonnull<const Value*>> {
    return elements_;
  }

  static auto classof(const Value* value) -> bool {
    return value->kind() == Kind::TupleValue ||
           value->kind() == Kind::TupleType;
  }

  template <typename F>
  auto Decompose(F f) const {
    return f(elements_);
  }

 private:
  std::vector<Nonnull<const Value*>> elements_;
};

// A tuple value.
class TupleValue : public TupleValueBase {
 public:
  // An empty tuple.
  static auto Empty() -> Nonnull<const TupleValue*> {
    static const TupleValue empty =
        TupleValue(std::vector<Nonnull<const Value*>>());
    return static_cast<Nonnull<const TupleValue*>>(&empty);
  }

  explicit TupleValue(std::vector<Nonnull<const Value*>> elements)
      : TupleValueBase(Kind::TupleValue, std::move(elements)) {}

  static auto classof(const Value* value) -> bool {
    return value->kind() == Kind::TupleValue;
  }
};

// A tuple type. These values are produced by converting a tuple value
// containing only types to type `type`.
class TupleType : public TupleValueBase {
 public:
  // The unit type.
  static auto Empty() -> Nonnull<const TupleType*> {
    static const TupleType empty =
        TupleType(std::vector<Nonnull<const Value*>>());
    return static_cast<Nonnull<const TupleType*>>(&empty);
  }

  explicit TupleType(std::vector<Nonnull<const Value*>> elements)
      : TupleValueBase(Kind::TupleType, std::move(elements)) {}

  static auto classof(const Value* value) -> bool {
    return value->kind() == Kind::TupleType;
  }
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

  template <typename F>
  auto Decompose(F f) const {
    return value_node_ ? f(*value_node_) : f();
  }

  auto value_node() const -> const std::optional<ValueNodeView>& {
    return value_node_;
  }

 private:
  std::optional<ValueNodeView> value_node_;
};

// Value for addr pattern
class AddrValue : public Value {
 public:
  explicit AddrValue(Nonnull<const Value*> pattern)
      : Value(Kind::AddrValue), pattern_(pattern) {}

  static auto classof(const Value* value) -> bool {
    return value->kind() == Kind::AddrValue;
  }

  template <typename F>
  auto Decompose(F f) const {
    return f(pattern_);
  }

  auto pattern() const -> const Value& { return *pattern_; }

 private:
  Nonnull<const Value*> pattern_;
};

// Value for uninitialized local variables.
class UninitializedValue : public Value {
 public:
  explicit UninitializedValue(Nonnull<const Value*> pattern)
      : Value(Kind::UninitializedValue), pattern_(pattern) {}

  static auto classof(const Value* value) -> bool {
    return value->kind() == Kind::UninitializedValue;
  }

  template <typename F>
  auto Decompose(F f) const {
    return f(pattern_);
  }

  auto pattern() const -> const Value& { return *pattern_; }

 private:
  Nonnull<const Value*> pattern_;
};

// The int type.
class IntType : public Value {
 public:
  IntType() : Value(Kind::IntType) {}

  static auto classof(const Value* value) -> bool {
    return value->kind() == Kind::IntType;
  }

  template <typename F>
  auto Decompose(F f) const {
    return f();
  }
};

// The bool type.
class BoolType : public Value {
 public:
  BoolType() : Value(Kind::BoolType) {}

  static auto classof(const Value* value) -> bool {
    return value->kind() == Kind::BoolType;
  }

  template <typename F>
  auto Decompose(F f) const {
    return f();
  }
};

// A type type.
class TypeType : public Value {
 public:
  TypeType() : Value(Kind::TypeType) {}

  static auto classof(const Value* value) -> bool {
    return value->kind() == Kind::TypeType;
  }

  template <typename F>
  auto Decompose(F f) const {
    return f();
  }
};

// A function type.
class FunctionType : public Value {
 public:
  // An explicit function parameter that is a `:!` binding:
  //
  //     fn MakeEmptyVector(T:! type) -> Vector(T);
  struct GenericParameter : public HashFromDecompose<GenericParameter> {
    template <typename F>
    auto Decompose(F f) const {
      return f(index, binding);
    }

    size_t index;
    Nonnull<const GenericBinding*> binding;
  };

  // For methods with unbound `self` parameters.
  struct MethodSelf : public HashFromDecompose<MethodSelf> {
    template <typename F>
    auto Decompose(F f) const {
      return f(addr_self, self_type);
    }

    // True if `self` parameter uses an `addr` pattern.
    bool addr_self;
    // Type of `self` parameter.
    const Value* self_type;
  };

  FunctionType(std::optional<MethodSelf> method_self,
               Nonnull<const Value*> parameters,
               Nonnull<const Value*> return_type)
      : FunctionType(method_self, parameters, {}, return_type, {}, {},
                     /*is_initializing=*/false) {}

  FunctionType(std::optional<MethodSelf> method_self,
               Nonnull<const Value*> parameters,
               std::vector<GenericParameter> generic_parameters,
               Nonnull<const Value*> return_type,
               std::vector<Nonnull<const GenericBinding*>> deduced_bindings,
               std::vector<Nonnull<const ImplBinding*>> impl_bindings,
               bool is_initializing)
      : Value(Kind::FunctionType),
        method_self_(method_self),
        parameters_(parameters),
        generic_parameters_(std::move(generic_parameters)),
        return_type_(return_type),
        deduced_bindings_(std::move(deduced_bindings)),
        impl_bindings_(std::move(impl_bindings)),
        is_initializing_(is_initializing) {}

  struct ExceptSelf : public HashFromDecompose<ExceptSelf> {
    template <typename F>
    auto Decompose(F f) const {
      return f();
    }
  };

  FunctionType(ExceptSelf, const FunctionType* clone)
      : FunctionType(std::nullopt, clone->parameters_,
                     clone->generic_parameters_, clone->return_type_,
                     clone->deduced_bindings_, clone->impl_bindings_,
                     clone->is_initializing_) {}

  static auto classof(const Value* value) -> bool {
    return value->kind() == Kind::FunctionType;
  }

  template <typename F>
  auto Decompose(F f) const {
    return f(method_self_, parameters_, generic_parameters_, return_type_,
             deduced_bindings_, impl_bindings_, is_initializing_);
  }

  // The type of the function parameter tuple.
  auto parameters() const -> const Value& { return *parameters_; }
  // Parameters that use a generic `:!` binding at the top level.
  auto generic_parameters() const -> llvm::ArrayRef<GenericParameter> {
    return generic_parameters_;
  }
  // The function return type.
  auto return_type() const -> const Value& { return *return_type_; }
  // All generic bindings in this function's signature that should be deduced
  // in a call. This excludes any generic parameters.
  auto deduced_bindings() const
      -> llvm::ArrayRef<Nonnull<const GenericBinding*>> {
    return deduced_bindings_;
  }
  // The bindings for the impl witness tables required by the
  // bounds on the type parameters of the generic function.
  auto impl_bindings() const -> llvm::ArrayRef<Nonnull<const ImplBinding*>> {
    return impl_bindings_;
  }
  // Return whether the function type is an initializing expression or not.
  auto is_initializing() const -> bool { return is_initializing_; }

  // Binding for the implicit `self` parameter, if this is an unbound method.
  auto method_self() const -> std::optional<MethodSelf> { return method_self_; }

 private:
  std::optional<MethodSelf> method_self_;
  Nonnull<const Value*> parameters_;
  std::vector<GenericParameter> generic_parameters_;
  Nonnull<const Value*> return_type_;
  std::vector<Nonnull<const GenericBinding*>> deduced_bindings_;
  std::vector<Nonnull<const ImplBinding*>> impl_bindings_;
  bool is_initializing_;
};

// A pointer type.
class PointerType : public Value {
 public:
  // Constructs a pointer type with the given pointee type.
  explicit PointerType(Nonnull<const Value*> pointee_type)
      : Value(Kind::PointerType), pointee_type_(pointee_type) {}

  static auto classof(const Value* value) -> bool {
    return value->kind() == Kind::PointerType;
  }

  template <typename F>
  auto Decompose(F f) const {
    return f(pointee_type_);
  }

  auto pointee_type() const -> const Value& { return *pointee_type_; }

 private:
  Nonnull<const Value*> pointee_type_;
};

// The `auto` type.
class AutoType : public Value {
 public:
  AutoType() : Value(Kind::AutoType) {}

  static auto classof(const Value* value) -> bool {
    return value->kind() == Kind::AutoType;
  }

  template <typename F>
  auto Decompose(F f) const {
    return f();
  }
};

// A struct type.
class StructType : public Value {
 public:
  StructType() : StructType(std::vector<NamedValue>{}) {}

  explicit StructType(std::vector<NamedValue> fields)
      : Value(Kind::StructType), fields_(std::move(fields)) {}

  static auto classof(const Value* value) -> bool {
    return value->kind() == Kind::StructType;
  }

  template <typename F>
  auto Decompose(F f) const {
    return f(fields_);
  }

  auto fields() const -> llvm::ArrayRef<NamedValue> { return fields_; }

 private:
  std::vector<NamedValue> fields_;
};

// A class type.
class NominalClassType : public Value {
 public:
  explicit NominalClassType(
      Nonnull<const ClassDeclaration*> declaration,
      Nonnull<const Bindings*> bindings,
      std::optional<Nonnull<const NominalClassType*>> base,
      Nonnull<const VTable*> class_vtable)
      : Value(Kind::NominalClassType),
        declaration_(declaration),
        bindings_(bindings),
        base_(base),
        vtable_(class_vtable),
        hierarchy_level_(base ? (*base)->hierarchy_level() + 1 : 0) {}

  static auto classof(const Value* value) -> bool {
    return value->kind() == Kind::NominalClassType;
  }

  template <typename F>
  auto Decompose(F f) const {
    return f(declaration_, bindings_, base_, vtable_);
  }

  auto declaration() const -> const ClassDeclaration& { return *declaration_; }

  auto bindings() const -> const Bindings& { return *bindings_; }

  auto base() const -> std::optional<Nonnull<const NominalClassType*>> {
    return base_;
  }

  auto type_args() const -> const BindingMap& { return bindings_->args(); }

  // Witnesses for each of the class's impl bindings.
  auto witnesses() const -> const ImplWitnessMap& {
    return bindings_->witnesses();
  }

  auto vtable() const -> const VTable& { return *vtable_; }

  // Returns how many levels from the top ancestor class it is. i.e. a class
  // with no base returns `0`, while a class with a `.base` and `.base.base`
  // returns `2`.
  auto hierarchy_level() const -> int { return hierarchy_level_; }

  // Returns whether this a parameterized class. That is, a class with
  // parameters and no corresponding arguments.
  auto IsParameterized() const -> bool {
    return declaration_->type_params().has_value() && type_args().empty();
  }

  // Returns whether this class is, or inherits `other`.
  auto InheritsClass(Nonnull<const Value*> other) const -> bool;

 private:
  Nonnull<const ClassDeclaration*> declaration_;
  Nonnull<const Bindings*> bindings_ = Bindings::None();
  const std::optional<Nonnull<const NominalClassType*>> base_;
  Nonnull<const VTable*> vtable_;
  int hierarchy_level_;
};

class MixinPseudoType : public Value {
 public:
  explicit MixinPseudoType(Nonnull<const MixinDeclaration*> declaration)
      : Value(Kind::MixinPseudoType), declaration_(declaration) {
    CARBON_CHECK(!declaration->params().has_value())
        << "missing arguments for parameterized mixin type";
  }
  explicit MixinPseudoType(Nonnull<const MixinDeclaration*> declaration,
                           Nonnull<const Bindings*> bindings)
      : Value(Kind::MixinPseudoType),
        declaration_(declaration),
        bindings_(bindings) {}

  static auto classof(const Value* value) -> bool {
    return value->kind() == Kind::MixinPseudoType;
  }

  template <typename F>
  auto Decompose(F f) const {
    return f(declaration_, bindings_);
  }

  auto declaration() const -> const MixinDeclaration& { return *declaration_; }

  auto bindings() const -> const Bindings& { return *bindings_; }

  auto args() const -> const BindingMap& { return bindings_->args(); }

  auto witnesses() const -> const ImplWitnessMap& {
    return bindings_->witnesses();
  }

  auto FindFunction(const std::string_view& name) const
      -> std::optional<Nonnull<const FunctionValue*>>;

 private:
  Nonnull<const MixinDeclaration*> declaration_;
  Nonnull<const Bindings*> bindings_ = Bindings::None();
};

// Returns the value of the function named `name` in this class, or
// nullopt if there is no such function.
auto FindFunction(std::string_view name,
                  llvm::ArrayRef<Nonnull<Declaration*>> members)
    -> std::optional<Nonnull<const FunctionValue*>>;

// Returns the value of the function named `name` in this class and its
// parents, or nullopt if there is no such function.
auto FindFunctionWithParents(std::string_view name,
                             const ClassDeclaration& class_decl)
    -> std::optional<Nonnull<const FunctionValue*>>;

// Return the declaration of the member with the given name.
auto FindMember(std::string_view name,
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
                         Nonnull<const Bindings*> bindings)
      : Value(Kind::InterfaceType),
        declaration_(declaration),
        bindings_(bindings) {}

  static auto classof(const Value* value) -> bool {
    return value->kind() == Kind::InterfaceType;
  }

  template <typename F>
  auto Decompose(F f) const {
    return f(declaration_, bindings_);
  }

  auto declaration() const -> const InterfaceDeclaration& {
    return *declaration_;
  }

  auto bindings() const -> const Bindings& { return *bindings_; }

  auto args() const -> const BindingMap& { return bindings_->args(); }

  auto witnesses() const -> const ImplWitnessMap& {
    return bindings_->witnesses();
  }

 private:
  Nonnull<const InterfaceDeclaration*> declaration_;
  Nonnull<const Bindings*> bindings_ = Bindings::None();
};

// A named constraint type.
class NamedConstraintType : public Value {
 public:
  explicit NamedConstraintType(
      Nonnull<const ConstraintDeclaration*> declaration,
      Nonnull<const Bindings*> bindings)
      : Value(Kind::NamedConstraintType),
        declaration_(declaration),
        bindings_(bindings) {}

  static auto classof(const Value* value) -> bool {
    return value->kind() == Kind::NamedConstraintType;
  }

  template <typename F>
  auto Decompose(F f) const {
    return f(declaration_, bindings_);
  }

  auto declaration() const -> const ConstraintDeclaration& {
    return *declaration_;
  }

  auto bindings() const -> const Bindings& { return *bindings_; }

 private:
  Nonnull<const ConstraintDeclaration*> declaration_;
  Nonnull<const Bindings*> bindings_ = Bindings::None();
};

// A constraint that requires implementation of an interface.
struct ImplsConstraint : public HashFromDecompose<ImplsConstraint> {
  template <typename F>
  auto Decompose(F f) const {
    return f(type, interface);
  }

  // The type that is required to implement the interface.
  Nonnull<const Value*> type;
  // The interface that is required to be implemented.
  Nonnull<const InterfaceType*> interface;
};

// A constraint that requires an intrinsic property of a type.
struct IntrinsicConstraint : public HashFromDecompose<IntrinsicConstraint>,
                             public Printable<IntrinsicConstraint> {
  enum Kind {
    // `type` intrinsically implicitly converts to `parameters[0]`.
    // TODO: Split ImplicitAs into more specific constraints (such as
    // derived-to-base pointer conversions).
    ImplicitAs,
  };

  explicit IntrinsicConstraint(Nonnull<const Value*> type, Kind kind,
                               std::vector<Nonnull<const Value*>> arguments)
      : type(type), kind(kind), arguments(std::move(arguments)) {}

  template <typename F>
  auto Decompose(F f) const {
    return f(type, kind, arguments);
  }

  // Print the intrinsic constraint.
  void Print(llvm::raw_ostream& out) const;

  // The type that is required to satisfy the intrinsic property.
  Nonnull<const Value*> type;
  // The kind of the intrinsic property.
  Kind kind;
  // Arguments for the intrinsic property. The meaning of these depends on
  // `kind`.
  std::vector<Nonnull<const Value*>> arguments;
};

// A constraint that a collection of values are known to be the same.
struct EqualityConstraint : public HashFromDecompose<EqualityConstraint> {
  template <typename F>
  auto Decompose(F f) const {
    return f(values);
  }

  // Visit the values in this equality constraint that are a single step away
  // from the given value according to this equality constraint. That is: if
  // `value` is identical to a value in `values`, then call the visitor on all
  // values in `values` that are not identical to `value`. Otherwise, do not
  // call the visitor.
  //
  // Stops and returns `false` if any call to the visitor returns `false`,
  // otherwise returns `true`.
  auto VisitEqualValues(
      Nonnull<const Value*> value,
      llvm::function_ref<bool(Nonnull<const Value*>)> visitor) const -> bool;

  std::vector<Nonnull<const Value*>> values;
};

// A constraint indicating that access to an associated constant should be
// replaced by another value.
struct RewriteConstraint : public HashFromDecompose<RewriteConstraint> {
  template <typename F>
  auto Decompose(F f) const {
    return f(constant, unconverted_replacement, unconverted_replacement_type,
             converted_replacement);
  }

  // The associated constant value that is rewritten.
  Nonnull<const AssociatedConstant*> constant;
  // The replacement in its original type.
  Nonnull<const Value*> unconverted_replacement;
  // The type of the replacement.
  Nonnull<const Value*> unconverted_replacement_type;
  // The replacement after conversion to the type of the associated constant.
  Nonnull<const Value*> converted_replacement;
};

// A context in which we might look up a name.
struct LookupContext : public HashFromDecompose<LookupContext> {
  template <typename F>
  auto Decompose(F f) const {
    return f(context);
  }

  Nonnull<const Value*> context;
};

// A type-of-type for an unknown constrained type.
//
// These types are formed by the `&` operator that combines constraints and by
// `where` expressions.
//
// A constraint has three main properties:
//
// * A collection of (type, interface) pairs for interfaces that are known to
//   be implemented by a type satisfying the constraint.
// * A collection of (type, intrinsic) pairs for intrinsic properties that are
//   known to be satisfied by a type satisfying the constraint.
// * A collection of sets of values, typically associated constants, that are
//   known to be the same.
// * A collection of contexts in which member name lookups will be performed
//   for a type variable whose type is this constraint.
//
// Within these properties, the constrained type can be referred to with a
// `VariableType` naming the `self_binding`.
class ConstraintType : public Value {
 public:
  explicit ConstraintType(
      Nonnull<const GenericBinding*> self_binding,
      std::vector<ImplsConstraint> impls_constraints,
      std::vector<IntrinsicConstraint> intrinsic_constraints,
      std::vector<EqualityConstraint> equality_constraints,
      std::vector<RewriteConstraint> rewrite_constraints,
      std::vector<LookupContext> lookup_contexts)
      : Value(Kind::ConstraintType),
        self_binding_(self_binding),
        impls_constraints_(std::move(impls_constraints)),
        intrinsic_constraints_(std::move(intrinsic_constraints)),
        equality_constraints_(std::move(equality_constraints)),
        rewrite_constraints_(std::move(rewrite_constraints)),
        lookup_contexts_(std::move(lookup_contexts)) {}

  static auto classof(const Value* value) -> bool {
    return value->kind() == Kind::ConstraintType;
  }

  template <typename F>
  auto Decompose(F f) const {
    return f(self_binding_, impls_constraints_, intrinsic_constraints_,
             equality_constraints_, rewrite_constraints_, lookup_contexts_);
  }

  auto self_binding() const -> Nonnull<const GenericBinding*> {
    return self_binding_;
  }

  auto impls_constraints() const -> llvm::ArrayRef<ImplsConstraint> {
    return impls_constraints_;
  }

  auto intrinsic_constraints() const -> llvm::ArrayRef<IntrinsicConstraint> {
    return intrinsic_constraints_;
  }

  auto equality_constraints() const -> llvm::ArrayRef<EqualityConstraint> {
    return equality_constraints_;
  }

  auto rewrite_constraints() const -> llvm::ArrayRef<RewriteConstraint> {
    return rewrite_constraints_;
  }

  auto lookup_contexts() const -> llvm::ArrayRef<LookupContext> {
    return lookup_contexts_;
  }

  // Visit the values in that are a single step away from the given value
  // according to equality constraints in this constraint type, that is, the
  // values `v` that are not identical to `value` but for which we have a
  // `value == v` equality constraint in this constraint type.
  //
  // Stops and returns `false` if any call to the visitor returns `false`,
  // otherwise returns `true`.
  auto VisitEqualValues(
      Nonnull<const Value*> value,
      llvm::function_ref<bool(Nonnull<const Value*>)> visitor) const -> bool;

 private:
  Nonnull<const GenericBinding*> self_binding_;
  std::vector<ImplsConstraint> impls_constraints_;
  std::vector<IntrinsicConstraint> intrinsic_constraints_;
  std::vector<EqualityConstraint> equality_constraints_;
  std::vector<RewriteConstraint> rewrite_constraints_;
  std::vector<LookupContext> lookup_contexts_;
};

// A witness table.
class Witness : public Value {
 protected:
  explicit Witness(Value::Kind kind) : Value(kind) {}

 public:
  static auto classof(const Value* value) -> bool {
    return value->kind() == Kind::ImplWitness ||
           value->kind() == Kind::BindingWitness ||
           value->kind() == Kind::ConstraintWitness ||
           value->kind() == Kind::ConstraintImplWitness;
  }
};

// The witness table for an impl.
class ImplWitness : public Witness {
 public:
  // Construct a witness for an impl.
  explicit ImplWitness(Nonnull<const ImplDeclaration*> declaration,
                       Nonnull<const Bindings*> bindings)
      : Witness(Kind::ImplWitness),
        declaration_(declaration),
        bindings_(bindings) {}

  static auto classof(const Value* value) -> bool {
    return value->kind() == Kind::ImplWitness;
  }

  template <typename F>
  auto Decompose(F f) const {
    return f(declaration_, bindings_);
  }

  auto declaration() const -> const ImplDeclaration& { return *declaration_; }

  auto bindings() const -> const Bindings& { return *bindings_; }

  auto type_args() const -> const BindingMap& { return bindings_->args(); }

  auto witnesses() const -> const ImplWitnessMap& {
    return bindings_->witnesses();
  }

 private:
  Nonnull<const ImplDeclaration*> declaration_;
  Nonnull<const Bindings*> bindings_ = Bindings::None();
};

// The symbolic witness corresponding to an unresolved impl binding.
class BindingWitness : public Witness {
 public:
  // Construct a witness for an impl binding.
  explicit BindingWitness(Nonnull<const ImplBinding*> binding)
      : Witness(Kind::BindingWitness), binding_(binding) {}

  static auto classof(const Value* value) -> bool {
    return value->kind() == Kind::BindingWitness;
  }

  template <typename F>
  auto Decompose(F f) const {
    return f(binding_);
  }

  auto binding() const -> Nonnull<const ImplBinding*> { return binding_; }

 private:
  Nonnull<const ImplBinding*> binding_;
};

// A witness for a constraint type, expressed as a tuple of witnesses for the
// individual impls constraints in the constraint type.
class ConstraintWitness : public Witness {
 public:
  explicit ConstraintWitness(std::vector<Nonnull<const Witness*>> witnesses)
      : Witness(Kind::ConstraintWitness), witnesses_(std::move(witnesses)) {}

  static auto classof(const Value* value) -> bool {
    return value->kind() == Kind::ConstraintWitness;
  }

  template <typename F>
  auto Decompose(F f) const {
    return f(witnesses_);
  }

  auto witnesses() const -> llvm::ArrayRef<Nonnull<const Witness*>> {
    return witnesses_;
  }

 private:
  std::vector<Nonnull<const Witness*>> witnesses_;
};

// A witness for an impls constraint in a constraint type, expressed in terms of
// a symbolic witness for the constraint type.
class ConstraintImplWitness : public Witness {
 public:
  // Make a witness for the given impls_constraint of the given `ConstraintType`
  // witness. If we're indexing into a known tuple of witnesses, pull out the
  // element.
  static auto Make(Nonnull<Arena*> arena, Nonnull<const Witness*> witness,
                   int index) -> Nonnull<const Witness*> {
    CARBON_CHECK(!llvm::isa<ImplWitness>(witness))
        << "impl witness has no components to access";
    if (const auto* constraint_witness =
            llvm::dyn_cast<ConstraintWitness>(witness)) {
      return constraint_witness->witnesses()[index];
    }
    return arena->New<ConstraintImplWitness>(witness, index);
  }

  explicit ConstraintImplWitness(Nonnull<const Witness*> constraint_witness,
                                 int index)
      : Witness(Kind::ConstraintImplWitness),
        constraint_witness_(constraint_witness),
        index_(index) {
    CARBON_CHECK(!llvm::isa<ConstraintWitness>(constraint_witness))
        << "should have resolved element from constraint witness";
  }

  static auto classof(const Value* value) -> bool {
    return value->kind() == Kind::ConstraintImplWitness;
  }

  template <typename F>
  auto Decompose(F f) const {
    return f(constraint_witness_, index_);
  }

  // Get the witness for the complete `ConstraintType`.
  auto constraint_witness() const -> Nonnull<const Witness*> {
    return constraint_witness_;
  }

  // Get the index of the impls constraint within the constraint type.
  auto index() const -> int { return index_; }

 private:
  Nonnull<const Witness*> constraint_witness_;
  int index_;
};

// Allocate a `ConstraintImplWitness` using the custom `Make` function.
template <>
struct AllocateTrait<ConstraintImplWitness> {
  template <typename... Args>
  static auto New(Nonnull<Arena*> arena, Args&&... args)
      -> Nonnull<const Witness*> {
    return ConstraintImplWitness::Make(arena, std::forward<Args>(args)...);
  }
};

// A choice type.
class ChoiceType : public Value {
 public:
  ChoiceType(Nonnull<const ChoiceDeclaration*> declaration,
             Nonnull<const Bindings*> bindings)
      : Value(Kind::ChoiceType),
        declaration_(declaration),
        bindings_(bindings) {}

  static auto classof(const Value* value) -> bool {
    return value->kind() == Kind::ChoiceType;
  }

  template <typename F>
  auto Decompose(F f) const {
    return f(declaration_, bindings_);
  }

  auto bindings() const -> const Bindings& { return *bindings_; }

  auto type_args() const -> const BindingMap& { return bindings_->args(); }

  auto declaration() const -> const ChoiceDeclaration& { return *declaration_; }

  auto IsParameterized() const -> bool {
    return declaration_->type_params().has_value();
  }

 private:
  Nonnull<const ChoiceDeclaration*> declaration_;
  Nonnull<const Bindings*> bindings_;
};

// A variable type.
class VariableType : public Value {
 public:
  explicit VariableType(Nonnull<const GenericBinding*> binding)
      : Value(Kind::VariableType), binding_(binding) {}

  static auto classof(const Value* value) -> bool {
    return value->kind() == Kind::VariableType;
  }

  template <typename F>
  auto Decompose(F f) const {
    return f(binding_);
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

  template <typename F>
  auto Decompose(F f) const {
    return f(declaration_, params_);
  }

  auto declaration() const -> const Declaration& { return *declaration_; }
  auto params() const -> const TuplePattern& { return *params_; }

 private:
  Nonnull<const Declaration*> declaration_;
  Nonnull<const TuplePattern*> params_;
};

// The name of a member of a class or interface.
//
// These values are used to represent the second operand of a compound member
// access expression: `x.(A.B)`, and can also be the value of an alias
// declaration, but cannot be used in most other contexts.
class MemberName : public Value, public Printable<MemberName> {
 public:
  MemberName(std::optional<Nonnull<const Value*>> base_type,
             std::optional<Nonnull<const InterfaceType*>> interface,
             Nonnull<const NamedElement*> member)
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

  template <typename F>
  auto Decompose(F f) const {
    return f(base_type_, interface_, member_);
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
  auto member() const -> const NamedElement& { return *member_; }
  // The name of the member.
  auto name() const -> std::string_view { return member().name(); }

 private:
  std::optional<Nonnull<const Value*>> base_type_;
  std::optional<Nonnull<const InterfaceType*>> interface_;
  Nonnull<const NamedElement*> member_;
};

// A symbolic value representing an associated constant.
//
// This is a value of the form `A.B` or `A.B.C` or similar, where `A` is a
// `VariableType`.
class AssociatedConstant : public Value {
 public:
  explicit AssociatedConstant(
      Nonnull<const Value*> base, Nonnull<const InterfaceType*> interface,
      Nonnull<const AssociatedConstantDeclaration*> constant,
      Nonnull<const Witness*> witness)
      : Value(Kind::AssociatedConstant),
        base_(base),
        interface_(interface),
        constant_(constant),
        witness_(witness) {}

  static auto classof(const Value* value) -> bool {
    return value->kind() == Kind::AssociatedConstant;
  }

  template <typename F>
  auto Decompose(F f) const {
    return f(base_, interface_, constant_, witness_);
  }

  // The type for which we denote an associated constant.
  auto base() const -> const Value& { return *base_; }

  // The interface within which the constant was declared.
  auto interface() const -> const InterfaceType& { return *interface_; }

  // The associated constant whose value is being denoted.
  auto constant() const -> const AssociatedConstantDeclaration& {
    return *constant_;
  }

  // Witness within which the constant's value can be found.
  auto witness() const -> const Witness& { return *witness_; }

 private:
  Nonnull<const Value*> base_;
  Nonnull<const InterfaceType*> interface_;
  Nonnull<const AssociatedConstantDeclaration*> constant_;
  Nonnull<const Witness*> witness_;
};

// The String type.
class StringType : public Value {
 public:
  StringType() : Value(Kind::StringType) {}

  static auto classof(const Value* value) -> bool {
    return value->kind() == Kind::StringType;
  }

  template <typename F>
  auto Decompose(F f) const {
    return f();
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

  template <typename F>
  auto Decompose(F f) const {
    return f(value_);
  }

  auto value() const -> const std::string& { return value_; }

 private:
  std::string value_;
};

class TypeOfMixinPseudoType : public Value {
 public:
  explicit TypeOfMixinPseudoType(Nonnull<const MixinPseudoType*> class_type)
      : Value(Kind::TypeOfMixinPseudoType), mixin_type_(class_type) {}

  static auto classof(const Value* value) -> bool {
    return value->kind() == Kind::TypeOfMixinPseudoType;
  }

  template <typename F>
  auto Decompose(F f) const {
    return f(mixin_type_);
  }

  auto mixin_type() const -> const MixinPseudoType& { return *mixin_type_; }

 private:
  Nonnull<const MixinPseudoType*> mixin_type_;
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

  template <typename F>
  auto Decompose(F f) const {
    return f(name_);
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
  explicit TypeOfMemberName(Nonnull<const NamedElement*> member)
      : Value(Kind::TypeOfMemberName), member_(member) {}

  static auto classof(const Value* value) -> bool {
    return value->kind() == Kind::TypeOfMemberName;
  }

  template <typename F>
  auto Decompose(F f) const {
    return f(member_);
  }

  // TODO: consider removing this or moving it elsewhere in the AST,
  // since it's arguably part of the expression value rather than its type.
  auto member() const -> const NamedElement& { return *member_; }

 private:
  Nonnull<const NamedElement*> member_;
};

// The type of a namespace name.
//
// Such expressions can appear only as the target of an `alias` declaration or
// as the left-hand side of a simple member access expression.
class TypeOfNamespaceName : public Value {
 public:
  explicit TypeOfNamespaceName(
      Nonnull<const NamespaceDeclaration*> namespace_decl)
      : Value(Kind::TypeOfNamespaceName), namespace_decl_(namespace_decl) {}

  static auto classof(const Value* value) -> bool {
    return value->kind() == Kind::TypeOfNamespaceName;
  }

  template <typename F>
  auto Decompose(F f) const {
    return f(namespace_decl_);
  }

  auto namespace_decl() const -> Nonnull<const NamespaceDeclaration*> {
    return namespace_decl_;
  }

 private:
  Nonnull<const NamespaceDeclaration*> namespace_decl_;
};

// The type of a statically-sized array.
//
// Note that values of this type are represented as tuples.
class StaticArrayType : public Value {
 public:
  // Constructs a statically-sized array type with the given element type and
  // size.
  StaticArrayType(Nonnull<const Value*> element_type,
                  std::optional<size_t> size)
      : Value(Kind::StaticArrayType),
        element_type_(element_type),
        size_(size) {}

  static auto classof(const Value* value) -> bool {
    return value->kind() == Kind::StaticArrayType;
  }

  template <typename F>
  auto Decompose(F f) const {
    return f(element_type_, size_);
  }

  auto element_type() const -> const Value& { return *element_type_; }
  auto size() const -> size_t {
    CARBON_CHECK(has_size());
    return *size_;
  }
  auto has_size() const -> bool { return size_.has_value(); }

 private:
  Nonnull<const Value*> element_type_;
  std::optional<size_t> size_;
};

template <typename R, typename F>
auto Value::Visit(F f) const -> R {
  switch (kind()) {
#define CARBON_VALUE_KIND(kind) \
  case Kind::kind:              \
    return f(static_cast<const kind*>(this));
#include "explorer/ast/value_kinds.def"
  }
}

}  // namespace Carbon

#endif  // CARBON_EXPLORER_AST_VALUE_H_
