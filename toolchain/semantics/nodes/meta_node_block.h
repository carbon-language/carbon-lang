// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEMANTICS_NODES_META_NODE_BLOCK_H_
#define CARBON_TOOLCHAIN_SEMANTICS_NODES_META_NODE_BLOCK_H_

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "toolchain/semantics/nodes/meta_node.h"

namespace Carbon::Semantics {

// The standard structure for declaration and statement blocks.
template <typename MetaNodeT>
struct MetaNodeBlock {
 public:
  MetaNodeBlock(llvm::SmallVector<MetaNodeT, 0> nodes,
                llvm::StringMap<MetaNodeT> name_lookup)
      : nodes_(std::move(nodes)), name_lookup_(std::move(name_lookup)) {}

  auto nodes() const -> llvm::ArrayRef<MetaNodeT> { return nodes_; }
  auto name_lookup() const -> const llvm::StringMap<MetaNodeT>& {
    return name_lookup_;
  }

 protected:
  llvm::SmallVector<MetaNodeT, 0> nodes_;
  llvm::StringMap<MetaNodeT> name_lookup_;
};

using DeclarationBlock = MetaNodeBlock<Declaration>;
using StatementBlock = MetaNodeBlock<Statement>;

}  // namespace Carbon::Semantics

#endif  // CARBON_TOOLCHAIN_SEMANTICS_NODES_META_NODE_BLOCK_H_
