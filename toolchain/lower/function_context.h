// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_LOWER_FUNCTION_CONTEXT_H_
#define CARBON_TOOLCHAIN_LOWER_FUNCTION_CONTEXT_H_

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "toolchain/lower/file_context.h"
#include "toolchain/sem_ir/file.h"

namespace Carbon::Lower {

// Context and shared functionality for lowering handlers that produce an
// `llvm::Function` definition.
class FunctionContext {
 public:
  explicit FunctionContext(FileContext& file_context, llvm::Function* function,
                           llvm::raw_ostream* vlog_stream);

  // Returns a basic block corresponding to the start of the given semantics
  // block, and enqueues it for emission.
  auto GetBlock(SemIR::InstBlockId block_id) -> llvm::BasicBlock*;

  // If we have not yet allocated a `BasicBlock` for this `block_id`, set it to
  // `block`, and enqueue `block_id` for emission. Returns whether we set the
  // block.
  auto TryToReuseBlock(SemIR::InstBlockId block_id, llvm::BasicBlock* block)
      -> bool;

  // Builds LLVM IR for the sequence of instructions in `block_id`.
  auto LowerBlock(SemIR::InstBlockId block_id) -> void;

  // Returns a phi node corresponding to the block argument of the given basic
  // block.
  auto GetBlockArg(SemIR::InstBlockId block_id, SemIR::TypeId type_id)
      -> llvm::PHINode*;

  // Returns a value for the given instruction.
  auto GetValue(SemIR::InstId inst_id) -> llvm::Value* {
    // All builtins are types, with the same empty lowered value.
    if (inst_id.is_builtin()) {
      return GetTypeAsValue();
    }

    auto it = locals_.find(inst_id);
    if (it != locals_.end()) {
      return it->second;
    }
    return file_context_->GetGlobal(inst_id);
  }

  // Sets the value for the given instruction.
  auto SetLocal(SemIR::InstId inst_id, llvm::Value* value) {
    bool added = locals_.insert({inst_id, value}).second;
    CARBON_CHECK(added) << "Duplicate local insert: " << inst_id << " "
                        << sem_ir().insts().Get(inst_id);
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

  // Create a synthetic block that corresponds to no SemIR::InstBlockId. Such
  // a block should only ever have a single predecessor, and is used when we
  // need multiple `llvm::BasicBlock`s to model the linear control flow in a
  // single SemIR::File block.
  auto MakeSyntheticBlock() -> llvm::BasicBlock*;

  // Determine whether block is the most recently created synthetic block.
  auto IsCurrentSyntheticBlock(llvm::BasicBlock* block) -> bool {
    return synthetic_block_ == block;
  }

  // After emitting an initializer `init_id`, finishes performing the
  // initialization of `dest_id` from that initializer. This is a no-op if the
  // initialization was performed in-place, and otherwise performs a store or a
  // copy.
  auto FinishInit(SemIR::TypeId type_id, SemIR::InstId dest_id,
                  SemIR::InstId source_id) -> void;

  auto llvm_context() -> llvm::LLVMContext& {
    return file_context_->llvm_context();
  }
  auto llvm_module() -> llvm::Module& { return file_context_->llvm_module(); }
  auto builder() -> llvm::IRBuilderBase& { return builder_; }
  auto sem_ir() -> const SemIR::File& { return file_context_->sem_ir(); }

 private:
  // Custom instruction inserter for our IR builder. Automatically names
  // instructions.
  class Inserter : public llvm::IRBuilderDefaultInserter {
   public:
    explicit Inserter(const SemIR::InstNamer* inst_namer)
        : inst_namer_(inst_namer) {}

    // Sets the instruction we are currently emitting.
    void SetCurrentInstId(SemIR::InstId inst_id) { inst_id_ = inst_id; }

   private:
    auto InsertHelper(llvm::Instruction* inst, const llvm::Twine& name,
                      llvm::BasicBlock* block,
                      llvm::BasicBlock::iterator insert_pt) const
        -> void override;

    // The instruction namer.
    const SemIR::InstNamer* inst_namer_;

    // The current instruction ID.
    SemIR::InstId inst_id_ = SemIR::InstId::Invalid;
  };

  // Emits a value copy for type `type_id` from `source_id` to `dest_id`.
  // `source_id` must produce a value representation for `type_id`, and
  // `dest_id` must be a pointer to a `type_id` object.
  auto CopyValue(SemIR::TypeId type_id, SemIR::InstId source_id,
                 SemIR::InstId dest_id) -> void;

  // Context for the overall lowering process.
  FileContext* file_context_;

  // The IR function we're generating.
  llvm::Function* function_;

  llvm::IRBuilder<llvm::ConstantFolder, Inserter> builder_;

  // The optional vlog stream.
  llvm::raw_ostream* vlog_stream_;

  // Maps a function's SemIR::File blocks to lowered blocks.
  llvm::DenseMap<SemIR::InstBlockId, llvm::BasicBlock*> blocks_;

  // The synthetic block we most recently created. May be null if there is no
  // such block.
  llvm::BasicBlock* synthetic_block_ = nullptr;

  // Maps a function's SemIR::File instructions to lowered values.
  llvm::DenseMap<SemIR::InstId, llvm::Value*> locals_;
};

// Declare handlers for each SemIR::File instruction.
#define CARBON_SEM_IR_INST_KIND(Name)                                \
  auto Handle##Name(FunctionContext& context, SemIR::InstId inst_id, \
                    SemIR::Name inst) -> void;
#include "toolchain/sem_ir/inst_kind.def"

}  // namespace Carbon::Lower

#endif  // CARBON_TOOLCHAIN_LOWER_FUNCTION_CONTEXT_H_
