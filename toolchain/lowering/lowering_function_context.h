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

namespace Carbon::Lower {

// Context and shared functionality for lowering handlers that produce an
// `llvm::Function` definition.
class FunctionContext {
 public:
  explicit FunctionContext(FileContext& file_context, llvm::Function* function);

  // Returns a basic block corresponding to the start of the given semantics
  // block, and enqueues it for emission.
  auto GetBlock(SemIR::NodeBlockId block_id) -> llvm::BasicBlock*;

  // If we have not yet allocated a `BasicBlock` for this `block_id`, set it to
  // `block`, and enqueue `block_id` for emission. Returns whether we set the
  // block.
  auto TryToReuseBlock(SemIR::NodeBlockId block_id, llvm::BasicBlock* block)
      -> bool;

  // Returns a phi node corresponding to the block argument of the given basic
  // block.
  auto GetBlockArg(SemIR::NodeBlockId block_id, SemIR::TypeId type_id)
      -> llvm::PHINode*;

  // Returns a local (versus global) value for the given node.
  auto GetLocal(SemIR::NodeId node_id) -> llvm::Value* {
    // All builtins are types, with the same empty lowered value.
    if (node_id.index < SemIR::BuiltinKind::ValidCount) {
      return GetTypeAsValue();
    }

    auto it = locals_.find(node_id);
    CARBON_CHECK(it != locals_.end()) << "Missing local: " << node_id;
    return it->second;
  }

  // Returns a local (versus global) value for the given node in loaded state.
  // Loads will only be inserted on an as-needed basis.
  auto GetLocalLoaded(SemIR::NodeId node_id) -> llvm::Value*;

  // Sets the value for the given node.
  auto SetLocal(SemIR::NodeId node_id, llvm::Value* value) {
    bool added = locals_.insert({node_id, value}).second;
    CARBON_CHECK(added) << "Duplicate local insert: " << node_id;
  }

  // Returns the requested index into val based on whether val is a pointer
  // type.
  auto GetIndexFromStructOrArray(llvm::Type* llvm_type, llvm::Value* val,
                                 unsigned idx, const llvm::Twine& name)
      -> llvm::Value* {
    return val->getType()->isPointerTy()
               ? builder().CreateStructGEP(llvm_type, val, idx, name)
               : builder().CreateExtractValue(val, idx, name);
  }

  // Gets a callable's function.
  auto GetFunction(SemIR::FunctionId function_id) -> llvm::Function* {
    return file_context_->GetFunction(function_id);
  }

  // Returns a lowered type for the given type_id.
  auto GetType(SemIR::TypeId type_id) -> llvm::Type* {
    return file_context_->GetType(type_id);
  }

  // Returns a lowered value to use for a value of type `type`.
  auto GetTypeAsValue() -> llvm::Value* {
    return file_context_->GetTypeAsValue();
  }

  // Create a synthetic block that corresponds to no SemIR::NodeBlockId. Such
  // a block should only ever have a single predecessor, and is used when we
  // need multiple `llvm::BasicBlock`s to model the linear control flow in a
  // single SemIR::File block.
  auto CreateSyntheticBlock() -> llvm::BasicBlock*;

  // Determine whether block is the most recently created synthetic block.
  auto IsCurrentSyntheticBlock(llvm::BasicBlock* block) -> bool {
    return synthetic_block_ == block;
  }

  // After emitting an initializer `init_id`, finishes performing the
  // initialization of `dest_id` from that initializer. This is a no-op if the
  // initialization was performed in-place, and otherwise performs a store or a
  // copy.
  auto FinishInitialization(SemIR::TypeId type_id, SemIR::NodeId dest_id,
                            SemIR::NodeId init_id) -> void;

  auto llvm_context() -> llvm::LLVMContext& {
    return file_context_->llvm_context();
  }
  auto llvm_module() -> llvm::Module& { return file_context_->llvm_module(); }
  auto builder() -> llvm::IRBuilder<>& { return builder_; }
  auto semantics_ir() -> const SemIR::File& {
    return file_context_->semantics_ir();
  }

 private:
  // Emits a value copy for type `type_id` from `source_id` to `dest_id`.
  // `source_id` must produce a value representation for `type_id`, and
  // `dest_id` must be a pointer to a `type_id` object.
  auto CopyValue(SemIR::TypeId type_id, SemIR::NodeId source_id,
                 SemIR::NodeId dest_id) -> void;

  // Context for the overall lowering process.
  FileContext* file_context_;

  // The IR function we're generating.
  llvm::Function* function_;

  llvm::IRBuilder<> builder_;

  // Maps a function's SemIR::File blocks to lowered blocks.
  llvm::DenseMap<SemIR::NodeBlockId, llvm::BasicBlock*> blocks_;

  // The synthetic block we most recently created. May be null if there is no
  // such block.
  llvm::BasicBlock* synthetic_block_ = nullptr;

  // Maps a function's SemIR::File nodes to lowered values.
  // TODO: Handle nested scopes. Right now this is just cleared at the end of
  // every block.
  llvm::DenseMap<SemIR::NodeId, llvm::Value*> locals_;
};

// Declare handlers for each SemIR::File node.
#define CARBON_SEMANTICS_NODE_KIND(Name)                             \
  auto Handle##Name(FunctionContext& context, SemIR::NodeId node_id, \
                    SemIR::Node node)                                \
      ->void;
#include "toolchain/semantics/semantics_node_kind.def"

}  // namespace Carbon::Lower

#endif  // CARBON_TOOLCHAIN_LOWERING_LOWERING_FUNCTION_CONTEXT_H_
