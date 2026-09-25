// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_LOWER_FUNCTION_CONTEXT_H_
#define CARBON_TOOLCHAIN_LOWER_FUNCTION_CONTEXT_H_

#include <type_traits>

#include "common/map.h"
#include "common/raw_string_ostream.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "toolchain/lower/file_context.h"
#include "toolchain/lower/specific_coalescer.h"
#include "toolchain/sem_ir/file.h"
#include "toolchain/sem_ir/ids.h"

namespace Carbon::Lower {

// Context and shared functionality for lowering handlers that produce an
// `llvm::Function` definition.
class FunctionContext {
 public:
  // `function` must not be null. `function_fingerprint` and `di_subprogram` may
  // be null (see members).
  explicit FunctionContext(
      FileContext& file_context, llvm::Function* function,
      FileContext& specific_file_context,
      SemIR::FunctionId specific_sem_ir_function_id,
      SemIR::SpecificId specific_id,
      SpecificCoalescer::SpecificFunctionFingerprint* function_fingerprint,
      llvm::DISubprogram* di_subprogram, llvm::raw_ostream* vlog_stream);

  // Describes a function's body fingerprint while creating the function body.
  // The final fingerprint is stored in the `FileContext` as a
  // `SpecificFunctionFingerprint`.
  //
  // Create two function fingerprints, where both fingerprints include data
  // that's evaluated (and hence lowered) differently based on the
  // `SpecificId`. `common_fingerprint` includes global values, types
  // and `FunctionId` for functions called inside the function body.
  // `specific_fingerprint` includes `SpecificId`s for functions called.
  //
  // For two specifics of the same generic:
  // - If `common_fingerprint` is different, the specifics cannot be coalesced.
  // - If `common_fingerprint` and `specific_fingerprint` are the
  //   same, the specifics can be coalesced without additional checks.
  // - If `common_fingerprint` is the same but `specific_fingerprint` is
  //   different, additional checks are needed, i.e. inspecting the non-hashed
  //   `SpecificId`s.
  //
  // TODO: Consider optimizations for repeated entries in both fingerprints.
  struct LoweringFunctionFingerprint {
    llvm::BLAKE3 common_fingerprint;
    llvm::BLAKE3 specific_fingerprint;
  };

  // A type in a particular file. This is used when lowering a specific and we
  // find a type that might be from the specific rather than the generic. By
  // convention, this represents a type that has not yet been added to the
  // specific fingerprint.
  struct TypeInFile {
    const SemIR::File* file;
    SemIR::TypeId type_id;

    auto GetPointeeType() -> TypeInFile {
      return {.file = file, .type_id = file->GetPointeeType(type_id)};
    }
  };

  // A value representation in a particular file. By convention, this represents
  // a value representation whose kind has been added to the fingerprint but
  // whose type has not.
  struct ValueReprInFile {
    const SemIR::File* file;
    SemIR::ValueRepr repr;

    auto type() -> TypeInFile {
      return {.file = file, .type_id = repr.type_id};
    }
  };

  // An inst in a particular file.
  struct InstInFile {
    const SemIR::File* file;
    SemIR::InstId inst_id;
  };

  // Returns a basic block corresponding to the start of the given semantics
  // block in the current specific.
  auto GetBlock(SemIR::InstBlockId block_id) -> llvm::BasicBlock*;

  // If we have not yet allocated a `BasicBlock` for this `block_id`, set it to
  // `block`. Returns whether we set the block.
  auto TryToReuseBlock(SemIR::InstBlockId block_id, llvm::BasicBlock* block)
      -> bool;

  // Builds LLVM IR for the sequence of instructions in `block_id`, in the
  // `BasicBlock` returned by `GetBlock`, which is moved to the end of the
  // function to keep the blocks in lexical order.
  auto LowerBlock(SemIR::InstBlockId block_id) -> void;

  // Builds LLVM IR for the sequence of instructions in `block_id`.
  auto LowerBlockContents(SemIR::InstBlockId block_id) -> void;

  // Builds LLVM IR for the specified instruction.
  auto LowerInst(SemIR::InstId inst_id) -> void;

  // Builds LLVM IR for a spliced branch instruction, such as an element of the
  // `SpliceBranches` operand, and for all blocks reachable from it. If the
  // instruction is a `SpecificInst`, the branch and the code it reaches are
  // lowered in that specific. Each spliced branch gets its own values and
  // blocks, but can refer to values from the enclosing code. A
  // `BranchNextIndex` in the spliced code branches to `next_block`.
  auto LowerSplicedBranch(SemIR::InstId inst_id, llvm::BasicBlock* next_block)
      -> void;

  // Returns the block that a `BranchNextIndex` in the current spliced code
  // branches to.
  auto GetNextIndexBlock() -> llvm::BasicBlock* {
    CARBON_CHECK(scope_->next_index_block,
                 "`BranchNextIndex` outside of spliced branch");
    return scope_->next_index_block;
  }

  // Returns a phi node corresponding to the block argument of the given basic
  // block.
  auto GetBlockArg(SemIR::InstBlockId block_id, TypeInFile type)
      -> llvm::PHINode*;

  // Returns the constant value of the given instruction in the current
  // specific, and the file that constant value is within.
  auto GetConstantValue(SemIR::InstId inst_id)
      -> std::pair<const SemIR::File*, SemIR::ConstantId>;

  // Returns whether the given instruction is treated as a constant in this
  // function. For template-dependent instructions this requires looking in the
  // specific, as we do not discover whether they are actually constant until
  // instantiation.
  auto IsConstant(SemIR::InstId) -> bool;

  // Returns a value for the given instruction.
  auto GetValue(SemIR::InstId inst_id) -> llvm::Value*;

  // Sets the value for the given instruction in the current specific.
  auto SetLocal(SemIR::InstId inst_id, llvm::Value* value) -> void {
    bool added = scope_->locals.Insert(inst_id, value).is_inserted();
    CARBON_CHECK(added, "Duplicate local insert: {0} {1}", inst_id,
                 sem_ir().insts().Get(inst_id));
  }

  // Returns a lowered type for the given type_id in the given file. This adds
  // the specified type to the fingerprint.
  auto GetType(TypeInFile type) -> llvm::Type* {
    auto* llvm_type = GetFileContext(type.file).GetType(type.type_id);
    AddTypeToCurrentFingerprint(llvm_type);
    return llvm_type;
  }

  // Returns the alignment of the given type_id. This adds the alignment to the
  // fingerprint.
  auto GetAlignment(TypeInFile type) -> llvm::Align {
    auto align = GetFileContext(type.file).GetAlignment(type.type_id);
    AddIntToCurrentFingerprint(align.value());
    return align;
  }

  // Returns the type of the given instruction in the current specific.
  auto GetTypeOfInst(SemIR::InstId inst_id) -> llvm::Type* {
    return GetType(GetTypeIdOfInst(inst_id));
  }

  // Returns the type of the given instruction in the current specific.
  auto GetTypeIdOfInst(SemIR::InstId inst_id) -> TypeInFile;

  // Returns the value representation of the given type. This adds the kind of
  // value representation, but not the underlying type, to the fingerprint.
  auto GetValueRepr(TypeInFile type) -> ValueReprInFile;

  // Returns the initializing representation of the given type. This adds the
  // kind of initializing representation to the fingerprint.
  auto GetInitRepr(TypeInFile type) -> SemIR::InitRepr;

  // Returns a lowered value to use for a value of type `type`.
  auto GetTypeAsValue() -> llvm::Value* {
    return file_context_->GetTypeAsValue();
  }

  // Returns a lowered value to use for a value of literal type.
  auto GetLiteralAsValue() -> llvm::Constant* {
    return file_context_->GetLiteralAsValue();
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

  // Creates an alloca instruction of the given type, adds it to the entry
  // block, and starts the lifetime of the corresponding storage.
  auto CreateAlloca(TypeInFile type, const llvm::Twine& name = llvm::Twine())
      -> llvm::AllocaInst*;

  // Returns the debug location to associate with the specified instruction.
  auto GetDebugLoc(SemIR::InstId inst_id) -> llvm::DebugLoc;

  // Emits a load of an object representation of type `type`.
  auto LoadObject(TypeInFile type, llvm::Value* addr, llvm::Twine name = "")
      -> llvm::Value*;

  // Emits a store of an object representation of type `type`.
  auto StoreObject(TypeInFile type, llvm::Value* value, llvm::Value* addr)
      -> void;

  // Emits the instructions necessary to initialize the storage at `dest_id`
  // from the repr-initializing expression `source_id`. This assumes the
  // instructions for `source_id` have already been emitted, so it's a no-op if
  // the initialization was performed in-place, and otherwise performs a store
  // or a copy.
  auto InitializeStorage(TypeInFile type, SemIR::InstId dest_id,
                         SemIR::InstId source_id) -> void;

  // Emits the instructions necessary to perform the initialization described by
  // `init_id` in-place in its storage.
  auto InitializeStorage(SemIR::InstId init_id) -> void;

  // When fingerprinting for a specific, adds the call, found in the function
  // body, to <function_id, specific_id>. `function_id` and `specific_id` are
  // IDs within the file identified by `function_file_id`.
  auto AddCallToCurrentFingerprint(SemIR::CheckIRId file_id,
                                   SemIR::FunctionId function_id,
                                   SemIR::SpecificId specific_id) -> void;

  // When fingerprinting for a specific, adds an integer.
  auto AddIntToCurrentFingerprint(uint64_t value) -> void;

  // When fingerprinting for a specific, adds an enumerator value.
  template <typename T>
    requires(std::is_enum_v<T>)
  auto AddEnumToCurrentFingerprint(T value) -> void {
    AddIntToCurrentFingerprint(static_cast<uint64_t>(value));
  }

  // When fingerprinting for a specific, adds the type.
  auto AddTypeToCurrentFingerprint(llvm::Type* type) -> void;

  // When fingerprinting for a specific of a template, adds the given
  // instruction.
  auto AddInstToCurrentFingerprint(SemIR::InstId inst_id) -> void;

  // Emits the final function fingerprints. Only called when function lowering
  // is complete.
  auto EmitFinalFingerprint() -> void;

  // Returns the FileContext to use for lowering in the given file.
  auto GetFileContext(const SemIR::File* file) -> FileContext& {
    // Avoid hash table lookup for the expected files.
    if (file == &sem_ir()) {
      return *file_context_;
    }
    if (file == &specific_sem_ir()) {
      return *specific_file_context_;
    }
    return file_context_->context().GetFileContext(file);
  }

  auto llvm_context() -> llvm::LLVMContext& {
    return file_context_->llvm_context();
  }
  auto llvm_module() -> llvm::Module& { return file_context_->llvm_module(); }
  auto llvm_function() -> llvm::Function& { return *function_; }
  auto builder() -> llvm::IRBuilderBase& { return builder_; }
  auto sem_ir() -> const SemIR::File& { return file_context_->sem_ir(); }

  // The file context for the file that `specific_id()` is within.
  auto specific_file_context() -> FileContext& {
    return *specific_file_context_;
  }
  // The file that `specific_id()` is within.
  auto specific_sem_ir() -> const SemIR::File& {
    return specific_file_context_->sem_ir();
  }

  auto specific_sem_ir_function_id() -> SemIR::FunctionId {
    return specific_sem_ir_function_id_;
  }

  // The specific ID for the code that is being lowered. This is the specific
  // for the function, unless we are lowering spliced code from a nested
  // generic, such as the body of a pack expansion. Note that this is an ID from
  // `specific_sem_ir()`, not from `sem_ir()`.
  auto specific_id() -> SemIR::SpecificId { return scope_->specific_id; }

  // TODO: could template on BuiltinFunctionKind if more format
  // globals are eventually needed.
  auto printf_int_format_string() -> llvm::Value* {
    auto* format_string = file_context_->printf_int_format_string();
    if (!format_string) {
      format_string = builder().CreateGlobalString("%d\n", "printf.int.format");
      file_context_->SetPrintfIntFormatString(format_string);
    }
    return format_string;
  }

  auto GetVtable(SemIR::VtableId vtable_id, SemIR::SpecificId specific_id) const
      -> llvm::Constant* {
    return file_context_->GetVtable(vtable_id, specific_id);
  }

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

  // State for lowering the instructions of a particular specific within this
  // function. The function body is lowered in the outermost scope. Code that is
  // spliced into the function from a nested generic, such as the body of a pack
  // expansion, is lowered in a nested scope for the corresponding specific, so
  // that each splice gets its own values and blocks.
  struct SpecificScope {
    // The specific whose instructions are being lowered, if any. This is an ID
    // from `specific_sem_ir()`.
    SemIR::SpecificId specific_id;

    // The enclosing scope, or null for the function body. Values from the
    // enclosing scopes can be used in this scope.
    SpecificScope* parent = nullptr;

    // The block that a `BranchNextIndex` branches to, or null if not within a
    // spliced branch.
    llvm::BasicBlock* next_index_block = nullptr;

    // Maps SemIR::File blocks to lowered blocks.
    Map<SemIR::InstBlockId, llvm::BasicBlock*> blocks;

    // The keys of `blocks`, in the order in which they were added. This is used
    // to find the blocks to lower for spliced code.
    llvm::SmallVector<SemIR::InstBlockId> block_ids;

    // Maps SemIR::File instructions to lowered values.
    Map<SemIR::InstId, llvm::Value*> locals;
  };

  // Emits a value copy for type `type` from `source_id` to `dest_id`.
  // `source_id` must produce a value representation for `type`, and
  // `dest_id` must be a pointer to a `type` object.
  auto CopyValue(TypeInFile type, SemIR::InstId source_id,
                 SemIR::InstId dest_id) -> void;

  // Emits an object representation copy for type `type` from `source_id` to
  // `dest_id`. `source_id` and `dest_id` must produce pointers to `type`
  // objects.
  auto CopyObject(TypeInFile type, SemIR::InstId source_id,
                  SemIR::InstId dest_id) -> void;

  // Returns the specific to use to evaluate the given constant from
  // `sem_ir()`. This is the specific for the enclosing scope whose generic the
  // constant belongs to, which is usually the current scope.
  auto GetSpecificForConstant(SemIR::ConstantId const_id) -> SemIR::SpecificId;

  // When fingerprinting for a specific, adds the global.
  auto AddGlobalToCurrentFingerprint(llvm::Value* global) -> void;

  // Context for lowering in the file that contains this function's
  // instructions.
  FileContext* file_context_;

  // The IR function we're generating.
  llvm::Function* function_;

  // Context for lowering in the file that contains our specifics. Note that
  // this is a different file than the one referred to by `file_context_` if we
  // are lowering a specific that was generated for a generic function defined
  // in a different file.
  FileContext* specific_file_context_;

  // The function id of the function we're lowering.
  SemIR::FunctionId specific_sem_ir_function_id_;

  // Builder for creating code in this function. The insertion point is held at
  // the location of the current SemIR instruction.
  llvm::IRBuilder<llvm::ConstantFolder, Inserter> builder_;

  // The instruction after all allocas. This is used as the insert point for new
  // allocas.
  llvm::Instruction* after_allocas_ = nullptr;

  llvm::DISubprogram* di_subprogram_;

  // The optional vlog stream.
  llvm::raw_ostream* vlog_stream_;

  // This is initialized and populated while lowering a specific function.
  // When complete, this is used to complete the function_fingerprint_.
  LoweringFunctionFingerprint current_fingerprint_;

  // The accumulated fingerprint is owned by the FileContext and passed into
  // the FunctionContext. The function fingerprint is currently only built for
  // specific functions, otherwise, this will be nullptr.
  SpecificCoalescer::SpecificFunctionFingerprint* function_fingerprint_;

  // The synthetic block we most recently created. May be null if there is no
  // such block.
  llvm::BasicBlock* synthetic_block_ = nullptr;

  // The scope for the function body. Its specific is the function's specific,
  // if the function is a specific.
  SpecificScope function_scope_;

  // The scope that we are currently lowering instructions within.
  SpecificScope* scope_ = &function_scope_;
};

namespace Internal {
// Determines whether InstT should have a `HandleInst` function.
template <typename InstT>
constexpr bool HasHandleInst =
    // is_lowered() == false indicates lowering should never see this inst kind.
    InstT::Kind.is_lowered() &&
    // Constant instructions never need to be explicitly lowered.
    InstT::Kind.constant_kind() != SemIR::InstConstantKind::Always &&
    InstT::Kind.constant_kind() != SemIR::InstConstantKind::AlwaysUnique &&
    // Instructions that always produce types don't need a `HandleInst` even if
    // they're not constant, because `type` has an empty runtime representation
    // and we assume that `InstIsType::Always` implies a lack of side effects.
    InstT::Kind.is_type() != SemIR::InstIsType::Always;

// Computes the function type to use for HandleInst for InstT.
template <typename InstT>
using FunctionTypeForHandleInst = std::conditional_t<
    HasHandleInst<InstT>,
    auto(FunctionContext& context, SemIR::InstId inst_id, InstT inst)->void,
    auto()->void>;
}  // namespace Internal

// Explicitly delete the overload generated for non-lowered instructions.
// These all produce the same signature, so we only need to delete it once.
auto HandleInst() -> void = delete;

// Provides handlers for instructions that occur in a FunctionContext. This
// should be defined for instructions for which `Internal::HasHandleInst<InstT>`
// is true. See `FunctionContext::LowerInst` for how this is used.
//
// The signature of an overload is:
//
//   auto HandleInst(FunctionContext& context, SemIR::InstId inst_id,
//                   InstT inst) -> void;
#define CARBON_SEM_IR_INST_KIND(Name) \
  Internal::FunctionTypeForHandleInst<SemIR::Name> HandleInst;
#include "toolchain/sem_ir/inst_kind.def"

}  // namespace Carbon::Lower

#endif  // CARBON_TOOLCHAIN_LOWER_FUNCTION_CONTEXT_H_
