// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef TOOLCHAIN_SEMANTICS_NODES_PATTERN_BINDING_H_
#define TOOLCHAIN_SEMANTICS_NODES_PATTERN_BINDING_H_

#include "common/ostream.h"
#include "toolchain/parser/parse_tree.h"
#include "toolchain/semantics/nodes/declared_name.h"
#include "toolchain/semantics/nodes/literal.h"

namespace Carbon::Semantics {

// Semantic information for a literal.
class PatternBinding {
 public:
  explicit PatternBinding(ParseTree::Node node, DeclaredName name, Literal type)
      : node_(node), name_(name), type_(type) {}

  void Print(llvm::raw_ostream& out) const { out << name_ << ": " << type_; }

  auto node() const -> ParseTree::Node { return node_; }
  auto name() const -> const DeclaredName& { return name_; }
  auto type() const -> const Literal& { return type_; }

 private:
  ParseTree::Node node_;
  DeclaredName name_;
  Literal type_;
};

}  // namespace Carbon::Semantics

#endif  // TOOLCHAIN_SEMANTICS_NODES_PATTERN_BINDING_H_
