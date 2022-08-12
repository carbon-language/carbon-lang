// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_EXPLORER_AST_IMPL_BINDING_H_
#define CARBON_EXPLORER_AST_IMPL_BINDING_H_

#include <map>

#include "common/check.h"
#include "common/ostream.h"
#include "explorer/ast/ast_node.h"
#include "explorer/ast/pattern.h"
#include "explorer/ast/value_category.h"

namespace Carbon {

class Value;
class Expression;
class ImplBinding;

// `ImplBinding` plays the role of the parameter for passing witness
// tables to a generic. However, unlike regular parameters
// (`BindingPattern`) there is no explicit syntax that corresponds to
// an `ImplBinding`, so they are not created during parsing. Instances
// of `ImplBinding` are created during type checking, when processing
// a type parameter (a `GenericBinding`), or an `is` requirement in
// a `where` clause.
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
  void PrintID(llvm::raw_ostream& out) const override;

  // The binding for the type variable.
  auto type_var() const -> Nonnull<const GenericBinding*> { return type_var_; }
  // The interface being implemented.
  auto interface() const -> Nonnull<const Value*> { return iface_; }

  // Required for the ValueNode interface
  auto constant_value() const -> std::optional<Nonnull<const Value*>> {
    return std::nullopt;
  }
  auto symbolic_identity() const -> std::optional<Nonnull<const Value*>> {
    return symbolic_identity_;
  }
  void set_symbolic_identity(Nonnull<const Value*> value) {
    CARBON_CHECK(!symbolic_identity_.has_value());
    symbolic_identity_ = value;
  }

  // The static type of the impl. Cannot be called before typechecking.
  auto static_type() const -> const Value& { return **static_type_; }

  // Sets the static type of the impl. Can only be called once, during
  // typechecking.
  void set_static_type(Nonnull<const Value*> type) {
    CARBON_CHECK(!static_type_.has_value());
    static_type_ = type;
  }
  auto value_category() const -> ValueCategory { return ValueCategory::Let; }

  // Return the original impl binding.
  auto original() const -> Nonnull<const ImplBinding*> {
    if (original_.has_value())
      return *original_;
    else
      return this;
  }

  // Set the original impl binding.
  void set_original(Nonnull<const ImplBinding*> orig) { original_ = orig; }

 private:
  Nonnull<const GenericBinding*> type_var_;
  Nonnull<const Value*> iface_;
  std::optional<Nonnull<const Value*>> symbolic_identity_;
  std::optional<Nonnull<const Value*>> static_type_;
  std::optional<Nonnull<const ImplBinding*>> original_;
};

}  // namespace Carbon

#endif  // CARBON_EXPLORER_AST_IMPL_BINDING_H_
