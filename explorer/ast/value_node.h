// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_EXPLORER_AST_VALUE_NODE_H_
#define CARBON_EXPLORER_AST_VALUE_NODE_H_

#include <functional>
#include <optional>
#include <string_view>

#include "explorer/ast/ast_node.h"
#include "explorer/ast/value_category.h"
#include "explorer/common/nonnull.h"

namespace Carbon {

class Value;

// The placeholder name exposed by anonymous ValueNodes.
static constexpr std::string_view AnonymousName = "_";

// ImplementsValueNode is true if NodeType::ImplementsCarbonValueNode
// is valid and names a type, indicating that NodeType implements the
// ValueNode interface, defined below.

template <typename NodeType, typename = void>
static constexpr bool ImplementsValueNode = false;

// ValueNode is an interface implemented by AstNodes that can be associated
// with a value, such as declarations and bindings. The interface consists of
// the following methods:
//
// // Returns the constant associated with the node.
// // This is called by the interpreter, not the type checker.
// auto constant_value() const -> std::optional<Nonnull<const Value*>>;
//
// // Returns the symbolic compile-time identity of the node.
// // This is called by the type checker, not the interpreter.
// auto symbolic_identity() const -> std::optional<Nonnull<const Value*>>;
//
// // Returns the static type of an IdentifierExpression that names *this.
// auto static_type() const -> const Value&;
//
// // Returns the value category of an IdentifierExpression that names *this.
// auto value_category() const -> ValueCategory;
//
// // Print the node's identity (e.g. its name).
// void PrintID(llvm::raw_ostream& out) const;
//
// TODO: consider turning the above documentation into real code, as sketched
// at https://godbolt.org/z/186oEozhc

template <typename T>
static constexpr bool
    ImplementsValueNode<T, typename T::ImplementsCarbonValueNode> = true;

class ValueNodeView {
 public:
  template <typename NodeType,
            typename = std::enable_if_t<ImplementsValueNode<NodeType>>>
  // NOLINTNEXTLINE(google-explicit-constructor)
  ValueNodeView(Nonnull<const NodeType*> node)
      // Type-erase NodeType, retaining a pointer to the base class AstNode
      // and using std::function to encapsulate the ability to call
      // the derived class's methods.
      : base_(node),
        constant_value_(
            [](const AstNode& base) -> std::optional<Nonnull<const Value*>> {
              return llvm::cast<NodeType>(base).constant_value();
            }),
        symbolic_identity_(
            [](const AstNode& base) -> std::optional<Nonnull<const Value*>> {
              return llvm::cast<NodeType>(base).symbolic_identity();
            }),
        print_([](const AstNode& base, llvm::raw_ostream& out) -> void {
          // TODO: change this to print a summary of the node
          return llvm::cast<NodeType>(base).PrintID(out);
        }),
        static_type_([](const AstNode& base) -> const Value& {
          return llvm::cast<NodeType>(base).static_type();
        }),
        value_category_([](const AstNode& base) -> ValueCategory {
          return llvm::cast<NodeType>(base).value_category();
        }) {}

  ValueNodeView(const ValueNodeView&) = default;
  ValueNodeView(ValueNodeView&&) = default;
  auto operator=(const ValueNodeView&) -> ValueNodeView& = default;
  auto operator=(ValueNodeView&&) -> ValueNodeView& = default;

  // Returns `node` as an instance of the base class AstNode.
  auto base() const -> const AstNode& { return *base_; }

  // Returns node->constant_value()
  auto constant_value() const -> std::optional<Nonnull<const Value*>> {
    return constant_value_(*base_);
  }

  // Returns node->symbolic_identity()
  auto symbolic_identity() const -> std::optional<Nonnull<const Value*>> {
    return symbolic_identity_(*base_);
  }

  void Print(llvm::raw_ostream& out) const { print_(*base_, out); }

  // Returns node->static_type()
  auto static_type() const -> const Value& { return static_type_(*base_); }

  // Returns node->value_category()
  auto value_category() const -> ValueCategory {
    return value_category_(*base_);
  }

  friend auto operator==(const ValueNodeView& lhs, const ValueNodeView& rhs)
      -> bool {
    return lhs.base_ == rhs.base_;
  }

  friend auto operator!=(const ValueNodeView& lhs, const ValueNodeView& rhs)
      -> bool {
    return lhs.base_ != rhs.base_;
  }

  friend auto operator<(const ValueNodeView& lhs, const ValueNodeView& rhs)
      -> bool {
    return std::less<>()(lhs.base_, rhs.base_);
  }

 private:
  Nonnull<const AstNode*> base_;
  std::function<std::optional<Nonnull<const Value*>>(const AstNode&)>
      constant_value_;
  std::function<std::optional<Nonnull<const Value*>>(const AstNode&)>
      symbolic_identity_;
  std::function<void(const AstNode&, llvm::raw_ostream&)> print_;
  std::function<const Value&(const AstNode&)> static_type_;
  std::function<ValueCategory(const AstNode&)> value_category_;
};

}  // namespace Carbon

#endif  // CARBON_EXPLORER_AST_VALUE_NODE_H_
