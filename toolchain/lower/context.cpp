// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/lower/context.h"

#include "common/check.h"
#include "common/growing_range.h"
#include "common/raw_string_ostream.h"
#include "common/vlog.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"
#include "toolchain/lower/file_context.h"
#include "toolchain/sem_ir/inst_namer.h"

namespace Carbon::Lower {

Context::Context(
    llvm::LLVMContext* llvm_context,
    llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> fs, bool want_debug_info,
    const Parse::GetTreeAndSubtreesStore* tree_and_subtrees_getters,
    llvm::StringRef module_name, int total_ir_count,
    llvm::raw_ostream* vlog_stream)
    : llvm_context_(llvm_context),
      llvm_module_(std::make_unique<llvm::Module>(module_name, *llvm_context)),
      file_system_(std::move(fs)),
      di_builder_(*llvm_module_),
      di_compile_unit_(
          want_debug_info
              ? BuildDICompileUnit(module_name, *llvm_module_, di_builder_)
              : nullptr),
      tree_and_subtrees_getters_(tree_and_subtrees_getters),
      vlog_stream_(vlog_stream),
      total_ir_count_(total_ir_count) {}

auto Context::GetFileContext(const SemIR::File* file,
                             const SemIR::InstNamer* inst_namer)
    -> FileContext& {
  auto insert_result = file_contexts_.Insert(file->check_ir_id(), [&] {
    auto file_context =
        std::make_unique<FileContext>(*this, *file, inst_namer, vlog_stream_);
    file_context->PrepareToLower();
    return file_context;
  });
  return *insert_result.value();
}

auto Context::LowerPendingDefinitions() -> void {
  // Lower function definitions for generics.
  for (auto [file_context, function_id, specific_id] :
       GrowingRange(specific_function_definitions_)) {
    file_context->BuildFunctionDefinition(function_id, specific_id);
  }
}

auto Context::Finalize() && -> std::unique_ptr<llvm::Module> {
  LowerPendingDefinitions();

  file_contexts_.ForEach(
      [](auto, auto& file_context) { file_context->Finalize(); });

  return std::move(llvm_module_);
}

auto Context::BuildDICompileUnit(llvm::StringRef module_name,
                                 llvm::Module& llvm_module,
                                 llvm::DIBuilder& di_builder)
    -> llvm::DICompileUnit* {
  llvm_module.addModuleFlag(llvm::Module::Max, "Dwarf Version", 5);
  llvm_module.addModuleFlag(llvm::Module::Warning, "Debug Info Version",
                            llvm::DEBUG_METADATA_VERSION);
  // TODO: Include directory path in the compile_unit_file.
  llvm::DIFile* compile_unit_file = di_builder.createFile(module_name, "");
  // TODO: Introduce a new language code for Carbon. C works well for now since
  // it's something debuggers will already know/have support for at least.
  // Probably have to bump to C++ at some point for virtual functions,
  // templates, etc.
  return di_builder.createCompileUnit(llvm::dwarf::DW_LANG_C, compile_unit_file,
                                      "carbon",
                                      /*isOptimized=*/false, /*Flags=*/"",
                                      /*RV=*/0);
}

auto Context::GetLocForDI(SemIR::AbsoluteNodeId abs_node_id) -> LocForDI {
  const auto& tree_and_subtrees =
      tree_and_subtrees_getters().Get(abs_node_id.check_ir_id())();
  const auto& tokens = tree_and_subtrees.tree().tokens();

  if (abs_node_id.node_id().has_value()) {
    auto token =
        tree_and_subtrees.GetSubtreeTokenRange(abs_node_id.node_id()).begin;
    return {.filename = tokens.source().filename(),
            .line_number = tokens.GetLineNumber(token),
            .column_number = tokens.GetColumnNumber(token)};
  } else {
    return {.filename = tokens.source().filename(),
            .line_number = 0,
            .column_number = 0};
  }
}

}  // namespace Carbon::Lower
