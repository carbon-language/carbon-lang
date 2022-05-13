// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEMANTICS_NODES_META_NODE_BLOCK_H_
#define CARBON_TOOLCHAIN_SEMANTICS_NODES_META_NODE_BLOCK_H_

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "toolchain/semantics/nodes/meta_node.h"

namespace Carbon {
class SemanticsIR;
class SemanticsIRFactory;
}  // namespace Carbon

namespace Carbon::Semantics {

// The standard structure for declaration and statement blocks.
template <typename MetaNodeT>
struct MetaNodeBlock {
 public:
  auto nodes() const -> llvm::ArrayRef<Declaration> { return nodes_; }
  auto name_lookup() const -> const llvm::StringMap<Declaration>& {
    return name_lookup_;
  }

 protected:
  friend class Carbon::SemanticsIR;
  friend class Carbon::SemanticsIRFactory;

  // TODO: Switch to direct construction instead of incremental adds?
  void add_node(MetaNodeT node) { nodes_.push_back(node); }
  void add_named_node(std::tuple<llvm::StringRef, MetaNodeT> named_node) {
    auto& [name, node] = named_node;
    add_node(node);
    name_lookup_[name] = node;
  }

  llvm::SmallVector<MetaNodeT, 0> nodes_;
  llvm::StringMap<MetaNodeT> name_lookup_;
};

using DeclarationBlock = MetaNodeBlock<Declaration>;
using StatementBlock = MetaNodeBlock<Statement>;

}  // namespace Carbon::Semantics

#endif  // CARBON_TOOLCHAIN_SEMANTICS_NODES_META_NODE_BLOCK_H_
