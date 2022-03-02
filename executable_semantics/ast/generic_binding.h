// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXECUTABLE_SEMANTICS_AST_GENERIC_BINDING_H_
#define EXECUTABLE_SEMANTICS_AST_GENERIC_BINDING_H_

#include <map>

#include "common/check.h"
#include "common/ostream.h"
#include "executable_semantics/ast/ast_node.h"
#include "executable_semantics/ast/value_category.h"

namespace Carbon {

class Value;
class Expression;
class ImplBinding;

// TODO: expand the kinds of things that can be deduced parameters.
//   For now, only generic parameters are supported.
class GenericBinding : public AstNode {
 public:
  using ImplementsCarbonValueNode = void;

  GenericBinding(SourceLocation source_loc, std::string name,
                 Nonnull<Expression*> type)
      : AstNode(AstNodeKind::GenericBinding, source_loc),
        name_(std::move(name)),
        type_(type) {}

  void Print(llvm::raw_ostream& out) const override;

  static auto classof(const AstNode* node) -> bool {
    return InheritsFromGenericBinding(node->kind());
  }

  auto name() const -> const std::string& { return name_; }
  auto type() const -> const Expression& { return *type_; }
  auto type() -> Expression& { return *type_; }

  // The static type of the binding. Cannot be called before typechecking.
  auto static_type() const -> const Value& { return **static_type_; }

  // Sets the static type of the binding. Can only be called once, during
  // typechecking.
  void set_static_type(Nonnull<const Value*> type) {
    CHECK(!static_type_.has_value());
    static_type_ = type;
  }

  auto value_category() const -> ValueCategory { return ValueCategory::Let; }
  auto constant_value() const -> std::optional<Nonnull<const Value*>> {
    return constant_value_;
  }

  // Sets the value returned by constant_value(). Can only be called once,
  // during typechecking.
  void set_constant_value(Nonnull<const Value*> value) {
    CHECK(!constant_value_.has_value());
    constant_value_ = value;
  }

  // The impl binding associated with this type variable.
  auto impl_binding() const -> std::optional<Nonnull<const ImplBinding*>> {
    return impl_binding_;
  }
  // Set the impl binding.
  void set_impl_binding(Nonnull<const ImplBinding*> binding) {
    CHECK(!impl_binding_.has_value());
    impl_binding_ = binding;
  }

 private:
  std::string name_;
  Nonnull<Expression*> type_;
  std::optional<Nonnull<const Value*>> static_type_;
  std::optional<Nonnull<const Value*>> constant_value_;
  std::optional<Nonnull<const ImplBinding*>> impl_binding_;
};

using BindingMap =
    std::map<Nonnull<const GenericBinding*>, Nonnull<const Value*>>;

// The run-time counterpart of a `GenericBinding`.
//
// Once a generic binding has been declared, it can be used
// in two different ways: as a compile-time constant with a
// symbolic value (such as a `VariableType`), or as a run-time
// variable with a concrete value that is stored on the stack.
// An `ImplBinding` is used in contexts where the second
// interpretation is intended.
class ImplBinding : public AstNode {
 public:
  using ImplementsCarbonValueNode = void;

  ImplBinding(SourceLocation source_loc,
              Nonnull<const GenericBinding*> type_var,
              Nonnull<const Value*> iface)
      : AstNode(AstNodeKind::ImplBinding, source_loc),
        type_var_(type_var),
        iface_(iface) {}

  static auto classof(const AstNode* node) -> bool {
    return InheritsFromImplBinding(node->kind());
  }
  void Print(llvm::raw_ostream& out) const override;

  // The binding for the type variable.
  auto type_var() const -> Nonnull<const GenericBinding*> { return type_var_; }
  // The interface being implemented.
  auto interface() const -> Nonnull<const Value*> { return iface_; }

  // Required for the the ValueNode interface
  auto constant_value() const -> std::optional<Nonnull<const Value*>> {
    return std::nullopt;
  }

  // The static type of the impl. Cannot be called before typechecking.
  auto static_type() const -> const Value& { return **static_type_; }

  // Sets the static type of the impl. Can only be called once, during
  // typechecking.
  void set_static_type(Nonnull<const Value*> type) {
    CHECK(!static_type_.has_value());
    static_type_ = type;
  }
  auto value_category() const -> ValueCategory { return ValueCategory::Let; }

 private:
  Nonnull<const GenericBinding*> type_var_;
  Nonnull<const Value*> iface_;
  std::optional<Nonnull<const Value*>> static_type_;
};

}  // namespace Carbon

#endif  // EXECUTABLE_SEMANTICS_AST_GENERIC_BINDING_H_
