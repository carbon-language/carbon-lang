// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEMANTICS_NODES_PATTERN_BINDING_H_
#define CARBON_TOOLCHAIN_SEMANTICS_NODES_PATTERN_BINDING_H_

#include "common/ostream.h"
#include "toolchain/parser/parse_tree.h"
#include "toolchain/semantics/nodes/declared_name.h"
#include "toolchain/semantics/nodes/expression.h"

namespace Carbon::Semantics {

// Semantic information for a literal.
class PatternBinding {
 public:
  explicit PatternBinding(ParseTree::Node node, DeclaredName name,
                          Expression type)
      : node_(node), name_(name), type_(type) {}

  auto node() const -> ParseTree::Node { return node_; }
  auto name() const -> const DeclaredName& { return name_; }
  auto type() const -> const Expression& { return type_; }

 private:
  ParseTree::Node node_;
  DeclaredName name_;
  Expression type_;
};

}  // namespace Carbon::Semantics

#endif  // CARBON_TOOLCHAIN_SEMANTICS_NODES_PATTERN_BINDING_H_
