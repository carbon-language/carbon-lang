// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEMANTICS_NODES_FUNCTION_H_
#define CARBON_TOOLCHAIN_SEMANTICS_NODES_FUNCTION_H_

#include "common/ostream.h"
#include "llvm/ADT/SmallVector.h"
#include "toolchain/parser/parse_tree.h"
#include "toolchain/semantics/meta_node.h"
#include "toolchain/semantics/meta_node_block.h"
#include "toolchain/semantics/nodes/declared_name.h"
#include "toolchain/semantics/nodes/pattern_binding.h"

// TODO: StatementBlock has some circularity in its forward declarations that
// needs to be fixed. These includes are a workaround.
#include "toolchain/semantics/nodes/expression_statement.h"
#include "toolchain/semantics/nodes/return.h"

namespace Carbon::Semantics {

// Represents `fn name(params...) [-> return_expr] body`.
class Function {
 public:
  static constexpr DeclarationKind MetaNodeKind = DeclarationKind::Function;

  Function(ParseTree::Node node, DeclaredName name,
           llvm::SmallVector<PatternBinding, 0> params,
           llvm::Optional<Semantics::Expression> return_expr,
           StatementBlock body)
      : node_(node),
        name_(name),
        params_(std::move(params)),
        return_expr_(return_expr),
        body_(std::move(body)) {}

  auto node() const -> ParseTree::Node { return node_; }
  auto name() const -> const DeclaredName& { return name_; }
  auto params() const -> llvm::ArrayRef<PatternBinding> { return params_; }
  auto return_expr() const -> llvm::Optional<Semantics::Expression> {
    return return_expr_;
  }

  auto body() const -> const StatementBlock& { return body_; }

 private:
  // The FunctionDeclaration node.
  ParseTree::Node node_;

  // The function's name.
  DeclaredName name_;

  // Regular function parameters.
  llvm::SmallVector<PatternBinding, 0> params_;

  // The return expression.
  llvm::Optional<Semantics::Expression> return_expr_;

  StatementBlock body_;
};

}  // namespace Carbon::Semantics

#endif  // CARBON_TOOLCHAIN_SEMANTICS_NODES_FUNCTION_H_
