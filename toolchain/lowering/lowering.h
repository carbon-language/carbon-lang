// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_LOWERING_LOWERING_H_
#define CARBON_TOOLCHAIN_LOWERING_LOWERING_H_

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "toolchain/semantics/semantics_ir.h"
#include "toolchain/semantics/semantics_node.h"

namespace Carbon {

// Use LowerToLLVM rather than calling this directly.
//
// This carries state for lowering. `Run()` should only be called once, and
// handles the main execution.
class Lowering {
 public:
  explicit Lowering(llvm::LLVMContext& llvm_context,
                    llvm::StringRef module_name,
                    const SemanticsIR& semantics_ir);

  // Lowers the SemanticsIR to LLVM IR.
  auto Run() -> std::unique_ptr<llvm::Module>;

 private:
  // Provides DenseMapInfo for SemanticsNodeId.
  struct SemanticsNodeIdMapInfo {
    static inline auto getEmptyKey() -> SemanticsNodeId {
      return SemanticsNodeId(llvm::DenseMapInfo<int32_t>::getEmptyKey());
    }
    static inline auto getTombstoneKey() -> SemanticsNodeId {
      return SemanticsNodeId(llvm::DenseMapInfo<int32_t>::getTombstoneKey());
    }

    static auto getHashValue(const SemanticsNodeId& val) -> unsigned {
      return llvm::DenseMapInfo<int32_t>::getHashValue(val.index);
    }

    static auto isEqual(const SemanticsNodeId& lhs, const SemanticsNodeId& rhs)
        -> bool {
      return lhs == rhs;
    }
  };

  // Declare handlers for each SemanticsIR node.
#define CARBON_SEMANTICS_NODE_KIND(Name) \
  auto Handle##Name##Node(SemanticsNodeId node_id, SemanticsNode node)->void;
#include "toolchain/semantics/semantics_node_kind.def"

  // Runs lowering for a block.
  auto LowerBlock(SemanticsNodeBlockId block_id) -> void;

  // Returns a type for the given node.
  auto LowerNodeToType(SemanticsNodeId node_id) -> llvm::Type*;

  // State for building the LLVM IR.
  llvm::LLVMContext* llvm_context_;
  std::unique_ptr<llvm::Module> llvm_module_;
  llvm::IRBuilder<> builder_;

  // The input Semantics IR.
  const SemanticsIR* const semantics_ir_;

  // Blocks which we've observed and need to lower.
  llvm::SmallVector<std::pair<llvm::BasicBlock*, SemanticsNodeBlockId>>
      todo_blocks_;

  // A mapping of nodes to designated values.
  // TODO: It might be worth considering other approaches, or at least if we
  // stick with this we'll probably want to clean up nodes as they leave scope.
  // However, for now, it's handy to make things work.
  llvm::DenseMap<SemanticsNodeId, llvm::Value*, SemanticsNodeIdMapInfo>
      node_values_;
};

}  // namespace Carbon

#endif  // CARBON_TOOLCHAIN_LOWERING_LOWERING_H_
