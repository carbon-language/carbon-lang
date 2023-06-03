// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_EXPLORER_AST_STORAGE_H_
#define CARBON_EXPLORER_AST_STORAGE_H_

#include <optional>

#include "common/check.h"
#include "common/ostream.h"
#include "explorer/ast/ast_node.h"
#include "explorer/ast/clone_context.h"
#include "explorer/ast/expression_category.h"
#include "explorer/common/nonnull.h"
#include "explorer/common/source_location.h"
namespace Carbon {

class Value;

// TODO
class Storage : public AstNode {
 public:
  using ImplementsCarbonValueNode = void;

  explicit Storage(SourceLocation source_loc)
      : AstNode(AstNodeKind::Storage, source_loc) {}
  explicit Storage(CloneContext& context, const Storage& other)
      : AstNode(context, other) {}

  void Print(llvm::raw_ostream& out) const override;
  void PrintID(llvm::raw_ostream& out) const override;

  static auto classof(const AstNode* node) -> bool {
    return InheritsFromPattern(node->kind());
  }

  // Required for the ValueNode interface
  auto constant_value() const -> std::optional<Nonnull<const Value*>> {
    return std::nullopt;
  }
  auto symbolic_identity() const -> std::optional<Nonnull<const Value*>> {
    return std::nullopt;
  }

  auto static_type() const -> const Value& { return **static_type_; }
  void set_static_type(Nonnull<const Value*> type) {
    CARBON_CHECK(!static_type_.has_value());
    static_type_ = type;
  }

  auto expression_category() const -> ExpressionCategory {
    return ExpressionCategory::Initializing;
  }

 private:
  std::optional<Nonnull<const Value*>> static_type_;
};

}  // namespace Carbon

#endif  // CARBON_EXPLORER_AST_STORAGE_H_
