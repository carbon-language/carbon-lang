// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXECUTABLE_SEMANTICS_AST_RETURN_TARGET_H_
#define EXECUTABLE_SEMANTICS_AST_RETURN_TARGET_H_

#include <functional>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include "executable_semantics/ast/ast_node.h"
#include "executable_semantics/ast/return_term.h"
#include "executable_semantics/common/nonnull.h"

namespace Carbon {

class Block;

template <typename T, typename = void>
static constexpr bool ImplementsReturnTarget = false;

template <typename T>
static constexpr bool
    ImplementsReturnTarget<T, typename T::ImplementsCarbonReturnTarget> = true;

// Non-owning type-erased wrapper around a const NodeType* `node`, where
// NodeType implements the ReturnTarget interface.
class ReturnTargetView {
 public:
  template <typename NodeType,
            typename = std::enable_if_t<ImplementsReturnTarget<NodeType>>>
  ReturnTargetView(Nonnull<const NodeType*> node)
      // Type-erase NodeType, retaining a pointer to the base class AstNode
      // and using std::function to encapsulate the ability to call
      // the derived class's methods.
      : base_(node),
        return_term_([](const AstNode& base) -> const ReturnTerm& {
          return llvm::cast<NodeType>(base).return_term();
        }),
        return_term_mut_([](AstNode& base) -> ReturnTerm& {
          return llvm::cast<NodeType>(base).return_term();
        }),
        body_([](const AstNode& base) -> std::optional<Nonnull<const Block*>> {
          return llvm::cast<NodeType>(base).body();
        }) {}

  ReturnTargetView(const ReturnTargetView&) = default;
  ReturnTargetView(ReturnTargetView&&) = default;
  auto operator=(const ReturnTargetView&) -> ReturnTargetView& = default;
  auto operator=(ReturnTargetView&&) -> ReturnTargetView& = default;

  // Returns `node` as an instance of the base class AstNode.
  auto base() const -> const AstNode& { return *base_; }

  // Returns node->return_term()
  auto return_term() const -> const ReturnTerm& { return return_term_(*base_); }
  auto return_term() -> ReturnTerm& {
    return return_term_mut_(*(AstNode*)base_);
  }

  // Returns node->body()
  auto body() const -> std::optional<Nonnull<const Block*>> {
    return body_(*base_);
  }

  friend auto operator==(const ReturnTargetView& lhs,
                         const ReturnTargetView& rhs) -> bool {
    return lhs.base_ == rhs.base_;
  }

  friend auto operator!=(const ReturnTargetView& lhs,
                         const ReturnTargetView& rhs) -> bool {
    return lhs.base_ != rhs.base_;
  }

  friend auto operator<(const ReturnTargetView& lhs,
                        const ReturnTargetView& rhs) -> bool {
    return std::less<>()(lhs.base_, rhs.base_);
  }

 private:
  Nonnull<const AstNode*> base_;
  std::function<const ReturnTerm&(const AstNode&)> return_term_;
  std::function<ReturnTerm&(AstNode&)> return_term_mut_;
  std::function<std::optional<Nonnull<const Block*>>(const AstNode&)> body_;
};

}  // namespace Carbon

#endif  // EXECUTABLE_SEMANTICS_AST_RETURN_TARGET_H_
