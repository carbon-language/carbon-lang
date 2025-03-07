// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_LOWER_FUNCTION_CONTEXT_H_
#define CARBON_TOOLCHAIN_LOWER_FUNCTION_CONTEXT_H_

#include "common/map.h"
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
                           SemIR::SpecificId specific_id,
                           llvm::DISubprogram* di_subprogram,
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
  auto LowerBlockContents(SemIR::InstBlockId block_id) -> void;

  // Builds LLVM IR for the specified instruction.
  auto LowerInst(SemIR::InstId inst_id) -> void;

  // Returns a phi node corresponding to the block argument of the given basic
  // block.
  auto GetBlockArg(SemIR::InstBlockId block_id, SemIR::TypeId type_id)
      -> llvm::PHINode*;

  // Returns a value for the given instruction.
  auto GetValue(SemIR::InstId inst_id) -> llvm::Value* {
    // All builtins are types, with the same empty lowered value.
    if (SemIR::IsSingletonInstId(inst_id)) {
      return GetTypeAsValue();
    }

    if (auto result = locals_.Lookup(inst_id)) {
      return result.value();
    }

    if (auto result = file_context_->global_variables().Lookup(inst_id)) {
      return result.value();
    }

    return file_context_->GetGlobal(inst_id, specific_id_);
  }

  // Sets the value for the given instruction.
  auto SetLocal(SemIR::InstId inst_id, llvm::Value* value) -> void {
    bool added = locals_.Insert(inst_id, value).is_inserted();
    CARBON_CHECK(added, "Duplicate local insert: {0} {1}", inst_id,
                 sem_ir().insts().Get(inst_id));
  }

  // Gets a callable's function.
  auto GetFunction(SemIR::FunctionId function_id) -> llvm::Function* {
    return file_context_->GetFunction(function_id);
  }

  // Gets or creates a callable's function.
  auto GetOrCreateFunction(SemIR::FunctionId function_id,
                           SemIR::SpecificId specific_id) -> llvm::Function* {
    return file_context_->GetOrCreateFunction(function_id, specific_id);
  }

  // Returns a lowered type for the given type_id.
  auto GetType(SemIR::TypeId type_id) -> llvm::Type* {
    return file_context_->GetType(type_id);
  }

  // Returns a lowered value to use for a value of type `type`.
  auto GetTypeAsValue() -> llvm::Value* {
    return file_context_->GetTypeAsValue();
  }

  // Returns a lowered value to use for a value of int literal type.
  auto GetIntLiteralAsValue() -> llvm::Constant* {
    return file_context_->GetIntLiteralAsValue();
  }

  // Returns the instruction immediately after all the existing static allocas.
  // This is the insert point for future static allocas.
  auto GetInstructionAfterAllocas() const -> llvm::Instruction* {
    return after_allocas_;
  }

  // Sets the instruction after static allocas. This should be called once,
  // after the first alloca is created.
  auto SetInstructionAfterAllocas(llvm::Instruction* after_allocas) -> void {
    CARBON_CHECK(!after_allocas_);
    after_allocas_ = after_allocas;
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
  auto llvm_function() -> llvm::Function& { return *function_; }
  auto specific_id() -> SemIR::SpecificId { return specific_id_; }
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
    auto SetCurrentInstId(SemIR::InstId inst_id) -> void { inst_id_ = inst_id; }

   private:
    auto InsertHelper(llvm::Instruction* inst, const llvm::Twine& name,
                      llvm::BasicBlock::iterator insert_pt) const
        -> void override;

    // The instruction namer.
    const SemIR::InstNamer* inst_namer_;

    // The current instruction ID.
    SemIR::InstId inst_id_ = SemIR::InstId::None;
  };

  // Emits a value copy for type `type_id` from `source_id` to `dest_id`.
  // `source_id` must produce a value representation for `type_id`, and
  // `dest_id` must be a pointer to a `type_id` object.
  auto CopyValue(SemIR::TypeId type_id, SemIR::InstId source_id,
                 SemIR::InstId dest_id) -> void;

  // Emits an object representation copy for type `type_id` from `source_id` to
  // `dest_id`. `source_id` and `dest_id` must produce pointers to `type_id`
  // objects.
  auto CopyObject(SemIR::TypeId type_id, SemIR::InstId source_id,
                  SemIR::InstId dest_id) -> void;

  // Context for the overall lowering process.
  FileContext* file_context_;

  // The IR function we're generating.
  llvm::Function* function_;

  // The specific id, if the function is a specific.
  SemIR::SpecificId specific_id_;

  // Builder for creating code in this function. The insertion point is held at
  // the location of the current SemIR instruction.
  llvm::IRBuilder<llvm::ConstantFolder, Inserter> builder_;

  // The instruction after all allocas. This is used as the insert point for new
  // allocas.
  llvm::Instruction* after_allocas_ = nullptr;

  llvm::DISubprogram* di_subprogram_;

  // The optional vlog stream.
  llvm::raw_ostream* vlog_stream_;

  // Maps a function's SemIR::File blocks to lowered blocks.
  Map<SemIR::InstBlockId, llvm::BasicBlock*> blocks_;

  // The synthetic block we most recently created. May be null if there is no
  // such block.
  llvm::BasicBlock* synthetic_block_ = nullptr;

  // Maps a function's SemIR::File instructions to lowered values.
  Map<SemIR::InstId, llvm::Value*> locals_;
};

// Provides handlers for instructions that occur in a FunctionContext. Although
// this is declared for all instructions, it should only be defined for
// instructions which are non-constant and not always typed. See
// `FunctionContext::LowerInst` for how this is used.
#define CARBON_SEM_IR_INST_KIND(Name)                              \
  auto HandleInst(FunctionContext& context, SemIR::InstId inst_id, \
                  SemIR::Name inst) -> void;
#include "toolchain/sem_ir/inst_kind.def"

}  // namespace Carbon::Lower

#endif  // CARBON_TOOLCHAIN_LOWER_FUNCTION_CONTEXT_H_
