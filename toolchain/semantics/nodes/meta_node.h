// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEMANTICS_NODES_META_NODE_H_
#define CARBON_TOOLCHAIN_SEMANTICS_NODES_META_NODE_H_

#include <cstdint>

namespace Carbon {
class SemanticsIR;
}  // namespace Carbon

namespace Carbon::Testing {
class SemanticsIRForTest;
}  // namespace Carbon::Testing

namespace Carbon::Semantics {

// The standard structure for nodes which have multiple subtypes.
//
// This flyweight pattern is used so that each subtype can be stored in its own
// vector, minimizing memory consumption and heap fragmentation when large
// quantities are being created.
template <typename KindT>
class MetaNode {
 public:
  MetaNode() : MetaNode(KindT::Invalid, -1) {}

 private:
  friend class Carbon::SemanticsIR;
  friend class Carbon::Testing::SemanticsIRForTest;

  MetaNode(KindT kind, int32_t index) : kind_(kind), index_(index) {
    // TODO: kind_ and index_ are currently unused, this suppresses the
    // warning.
    kind_ = kind;
    index_ = index;
  }

  KindT kind_;

  // The index of the named entity within its list.
  int32_t index_;
};

enum class DeclarationKind {
  Invalid,
  Function,
};
using Declaration = MetaNode<DeclarationKind>;

enum class ExpressionKind {
  Invalid,
  Literal,
};
using Expression = MetaNode<ExpressionKind>;

enum class StatementKind {
  Invalid,
  Expression,
  Return,
};
using Statement = MetaNode<StatementKind>;

}  // namespace Carbon::Semantics

#endif  // CARBON_TOOLCHAIN_SEMANTICS_NODES_META_NODE_H_
