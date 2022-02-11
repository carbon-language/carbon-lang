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

// TODO: expand the kinds of things that can be deduced parameters.
//   For now, only generic parameters are supported.
class GenericBinding : public AstNode {
 public:
  using ImplementsCarbonNamedEntity = void;

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
  void set_static_type(Nonnull<const Value*> type) { static_type_ = type; }

  // Returns whether the static type has been set. Should only be called
  // during typechecking: before typechecking it's guaranteed to be false,
  // and after typechecking it's guaranteed to be true.
  auto has_static_type() const -> bool { return static_type_.has_value(); }

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
  // Unset the `constant_value()` after typechecking so that the
  // generic binding can be used during interpretation to bind the
  // witness table on the runtime stack.
  void unset_constant_value() { constant_value_ = std::nullopt; }

 private:
  std::string name_;
  Nonnull<Expression*> type_;
  std::optional<Nonnull<const Value*>> static_type_;
  std::optional<Nonnull<const Value*>> constant_value_;
};

typedef std::map<Nonnull<const GenericBinding*>, Nonnull<const Value*>>
    BindingMap;

}  // namespace Carbon

#endif  // EXECUTABLE_SEMANTICS_AST_GENERIC_BINDING_H_
