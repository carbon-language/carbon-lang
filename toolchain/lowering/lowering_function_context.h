// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_LOWERING_LOWERING_FUNCTION_CONTEXT_H_
#define CARBON_TOOLCHAIN_LOWERING_LOWERING_FUNCTION_CONTEXT_H_

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "toolchain/lowering/lowering_context.h"
#include "toolchain/semantics/semantics_ir.h"
#include "toolchain/semantics/semantics_node.h"

namespace Carbon {

// Context and shared functionality for lowering handlers that produce an
// `llvm::Function` definition.
class LoweringFunctionContext {
 public:
  explicit LoweringFunctionContext(LoweringContext& lowering_context,
                                   llvm::Function* function,
                                   llvm::raw_ostream* vlog_stream);

  // Lowers the given function to LLVM IR. Should only be called once.
  auto BuildFunctionDefinition(const SemanticsFunction& function) -> void;

  // Returns a local (versus global) value for the given node.
  auto GetLocal(SemanticsNodeId node_id) -> llvm::Value* {
    auto it = locals_.find(node_id);
    CARBON_CHECK(it != locals_.end()) << "Missing local: " << node_id;
    return it->second;
  }

  // Returns a local (versus global) value for the given node in loaded state.
  // Loads will only be inserted on an as-needed basis.
  auto GetLocalLoaded(SemanticsNodeId node_id) -> llvm::Value*;

  // Sets the value for the given node.
  auto SetLocal(SemanticsNodeId node_id, llvm::Value* value) {
    bool added = locals_.insert({node_id, value}).second;
    CARBON_CHECK(added) << "Duplicate local insert: " << node_id;
  }

  // Gets a callable's function.
  auto GetFunction(SemanticsFunctionId function_id) -> llvm::Function* {
    return lowering_context_->GetFunction(function_id);
  }

  // Returns a lowered type for the given type_id.
  auto GetType(SemanticsTypeId type_id) -> llvm::Type* {
    return lowering_context_->GetType(type_id);
  }

  auto lowering_context() -> LoweringContext& { return *lowering_context_; }
  auto llvm_context() -> llvm::LLVMContext& {
    return lowering_context_->llvm_context();
  }
  auto llvm_module() -> llvm::Module& {
    return lowering_context_->llvm_module();
  }
  auto builder() -> llvm::IRBuilder<>& { return builder_; }
  auto semantics_ir() -> const SemanticsIR& {
    return lowering_context_->semantics_ir();
  }

 private:
  // Context for the overall lowering process.
  LoweringContext* lowering_context_;

  // The IR function we're generating.
  llvm::Function* function_;

  llvm::IRBuilder<> builder_;

  // The optional vlog stream.
  llvm::raw_ostream* vlog_stream_;

  // Maps a function's SemanticsIR nodes to lowered values.
  // TODO: Handle nested scopes. Right now this is just cleared at the end of
  // every block.
  llvm::DenseMap<SemanticsNodeId, llvm::Value*> locals_;
};

// Declare handlers for each SemanticsIR node.
#define CARBON_SEMANTICS_NODE_KIND(Name)                                 \
  auto LoweringHandle##Name(LoweringFunctionContext& context,            \
                            SemanticsNodeId node_id, SemanticsNode node) \
      ->void;
#include "toolchain/semantics/semantics_node_kind.def"

}  // namespace Carbon

#endif  // CARBON_TOOLCHAIN_LOWERING_LOWERING_FUNCTION_CONTEXT_H_
