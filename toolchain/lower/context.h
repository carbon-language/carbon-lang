// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_LOWER_CONTEXT_H_
#define CARBON_TOOLCHAIN_LOWER_CONTEXT_H_

#include <memory>
#include <optional>
#include <utility>

#include "llvm/IR/Constants.h"
#include "llvm/IR/DIBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "toolchain/base/fixed_size_value_store.h"
#include "toolchain/parse/tree_and_subtrees.h"
#include "toolchain/sem_ir/absolute_node_id.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/inst_namer.h"

namespace Carbon::Lower {

class FileContext;

// Context for lowering to an LLVM module.
class Context {
 public:
  // Location information for use with DebugInfo. The line_number and
  // column_number are >= 0, with 0 as unknown, so that they can be passed
  // directly to DebugInfo.
  struct LocForDI {
    llvm::StringRef filename;
    int32_t line_number;
    int32_t column_number;
  };

  // A specific function whose definition needs to be lowered.
  struct PendingSpecificFunctionDefinition {
    FileContext* context;
    SemIR::FunctionId function_id;
    SemIR::SpecificId specific_id;
  };

  // `llvm_context` and `tree_and_subtrees_getters` must be non-null.
  // `vlog_stream` is optional.
  explicit Context(
      llvm::LLVMContext* llvm_context,
      llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> fs, bool want_debug_info,
      const Parse::GetTreeAndSubtreesStore* tree_and_subtrees_getters,
      llvm::StringRef module_name, int total_ir_count,
      llvm::raw_ostream* vlog_stream);

  // Gets or creates the `FileContext` for a given SemIR file. If an
  // `inst_namer` is specified the first time this is called for a file, it will
  // be used for that file. Otherwise, no instruction namer will be used.
  // TODO: Consider building an InstNamer if we're not given one.
  auto GetFileContext(const SemIR::File* file,
                      const SemIR::InstNamer* inst_namer = nullptr)
      -> FileContext&;

  // Registers a specific function definition to be lowered later.
  auto AddPendingSpecificFunctionDefinition(
      PendingSpecificFunctionDefinition pending) -> void {
    specific_function_definitions_.push_back(pending);
  }

  // Finishes lowering and takes ownership of the LLVM module. The context
  // cannot be used further after calling this.
  auto Finalize() && -> std::unique_ptr<llvm::Module>;

  // Returns location information for use with DebugInfo.
  auto GetLocForDI(SemIR::AbsoluteNodeId abs_node_id) -> LocForDI;

  // Returns a lowered value to use for a value of type `type`.
  auto GetTypeAsValue() -> llvm::Constant* {
    return llvm::ConstantStruct::get(GetTypeType());
  }

  // Returns a lowered value to use for a value of literal type.
  auto GetLiteralAsValue() -> llvm::Constant* {
    // TODO: Consider adding a named struct type for literals.
    return llvm::ConstantStruct::get(llvm::StructType::get(llvm_context()));
  }

  // Returns the empty LLVM struct type used to represent the type `type`.
  auto GetTypeType() -> llvm::StructType* {
    if (!type_type_) {
      // `type` is lowered to an empty LLVM StructType.
      type_type_ = llvm::StructType::create(*llvm_context_, {}, "type");
    }
    return type_type_;
  }

  auto llvm_context() -> llvm::LLVMContext& { return *llvm_context_; }
  auto llvm_module() -> llvm::Module& { return *llvm_module_; }
  auto file_system() -> llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem>& {
    return file_system_;
  }
  auto di_builder() -> llvm::DIBuilder& { return di_builder_; }
  auto di_compile_unit() -> llvm::DICompileUnit* { return di_compile_unit_; }
  auto tree_and_subtrees_getters() -> const Parse::GetTreeAndSubtreesStore& {
    return *tree_and_subtrees_getters_;
  }
  auto total_ir_count() -> int { return total_ir_count_; }

  auto printf_int_format_string() -> llvm::Value* {
    return printf_int_format_string_;
  }
  auto SetPrintfIntFormatString(llvm::Value* printf_int_format_string) {
    CARBON_CHECK(!printf_int_format_string_,
                 "PrintInt formatting string already generated");
    printf_int_format_string_ = printf_int_format_string;
  }

 private:
  // Create the DICompileUnit metadata for this compilation.
  auto BuildDICompileUnit(llvm::StringRef module_name,
                          llvm::Module& llvm_module,
                          llvm::DIBuilder& di_builder) -> llvm::DICompileUnit*;

  // Lower any definitions that have been registered for later lowering.
  // Currently, this lowers specifics for generic functions.
  auto LowerPendingDefinitions() -> void;

  // State for building the LLVM IR.
  llvm::LLVMContext* llvm_context_;
  std::unique_ptr<llvm::Module> llvm_module_;

  // The filesystem for source code.
  llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> file_system_;

  // State for building the LLVM IR debug info metadata.
  llvm::DIBuilder di_builder_;

  // The DICompileUnit, if any - null implies debug info is not being emitted.
  llvm::DICompileUnit* di_compile_unit_;

  // Parse trees. Used for debug information and crash diagnostics.
  const Parse::GetTreeAndSubtreesStore* tree_and_subtrees_getters_;

  // The optional vlog stream.
  llvm::raw_ostream* vlog_stream_;

  // The total number of files.
  int total_ir_count_;

  // The `FileContext`s for each IR that is involved in this lowering action.
  using FileContextStore =
      FixedSizeValueStore<SemIR::CheckIRId, std::unique_ptr<FileContext>>;
  FileContextStore file_contexts_;

  // Lowered version of the builtin type `type`.
  llvm::StructType* type_type_ = nullptr;

  // Global format string for `printf.int.format` used by the PrintInt builtin.
  llvm::Value* printf_int_format_string_ = nullptr;

  // Tracks which specific functions need to have their definitions lowered.
  // This list may grow while lowering generic definitions from this list.
  llvm::SmallVector<PendingSpecificFunctionDefinition>
      specific_function_definitions_;
};

}  // namespace Carbon::Lower

#endif  // CARBON_TOOLCHAIN_LOWER_CONTEXT_H_
