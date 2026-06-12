// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/lower/context.h"

#include "clang/Basic/SourceManager.h"
#include "clang/Sema/MultiplexExternalSemaSource.h"
#include "common/check.h"
#include "common/growing_range.h"
#include "common/raw_string_ostream.h"
#include "common/vlog.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"
#include "toolchain/lower/file_context.h"
#include "toolchain/sem_ir/inst_namer.h"
#include "toolchain/sem_ir/read_only_ast_source.h"

namespace Carbon::Lower {

Context::Context(
    llvm::LLVMContext* llvm_context,
    llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> fs, bool want_debug_info,
    const Parse::GetTreeAndSubtreesStore* tree_and_subtrees_getters,
    clang::CodeGenerator* clang_code_generator, llvm::StringRef module_name,
    int total_ir_count, Lower::OptimizationLevel opt_level,
    bool mangle_string_fingerprint, llvm::raw_ostream* vlog_stream)
    : llvm_context_(llvm_context),
      clang_code_generator_(clang_code_generator),
      llvm_module_owner_(
          clang_code_generator_
              ? nullptr
              : std::make_unique<llvm::Module>(module_name, *llvm_context)),
      llvm_module_(llvm_module_owner_ ? llvm_module_owner_.get()
                                      : clang_code_generator_->GetModule()),
      file_system_(std::move(fs)),
      opt_level_(opt_level),
      di_builder_(*llvm_module_),
      di_compile_unit_(want_debug_info
                           ? BuildDICompileUnit(llvm_module_->getName(),
                                                *llvm_module_, di_builder_)
                           : nullptr),
      tree_and_subtrees_getters_(tree_and_subtrees_getters),
      vlog_stream_(vlog_stream),
      total_ir_count_(total_ir_count),
      mangle_string_fingerprint_(mangle_string_fingerprint),
      file_contexts_(
          FileContextStore::MakeForOverwriteWithExplicitSize(total_ir_count_)) {
}

auto Context::GetFileContext(const SemIR::File* file,
                             const SemIR::InstNamer* inst_namer)
    -> FileContext& {
  auto& file_context = file_contexts_.Get(file->check_ir_id());
  if (!file_context) {
    file_context =
        std::make_unique<FileContext>(*this, *file, inst_namer, vlog_stream_);
    file_context->PrepareToLower();
  }
  return *file_context;
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

  for (auto& file_context : file_contexts_.values()) {
    if (file_context) {
      if (file_context->cpp_file()) {
        // Remove the `CarbonExternalASTSource` installed during check
        // (always the last child of the multiplex source) and replace
        // it with a `ReadOnlyASTSource`. This is necessary because the
        // original source has a now-invalid pointer to a
        // `Check::Context`.
        auto& ast = const_cast<clang::ASTContext&>(
            file_context->cpp_file()->ast_context());
        auto* multiplex_source =
            cast<clang::MultiplexExternalSemaSource>(ast.getExternalSource());
        auto& child_sources = multiplex_source->GetSources();
        child_sources.pop_back();
        multiplex_source->AddSource(
            llvm::makeIntrusiveRefCnt<SemIR::ReadOnlyASTSource>(
                file_context->sem_ir()));
      }

      file_context->Finalize();
    }
  }

  return clang_code_generator_ ? std::unique_ptr<llvm::Module>(
                                     clang_code_generator_->ReleaseModule())
                               : std::move(llvm_module_owner_);
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
  // TODO: Introduce a new language code for Carbon. C++ works well for now
  // since it's something debuggers will already know/have support for at least.
  return di_builder.createCompileUnit(llvm::dwarf::DW_LANG_C_plus_plus,
                                      compile_unit_file, "carbon",
                                      /*isOptimized=*/false, /*Flags=*/"",
                                      /*RV=*/0);
}

auto Context::GetLocForDI(SemIR::AbsoluteNodeRef abs_node_id) -> LocForDI {
  if (abs_node_id.is_cpp()) {
    const SemIR::File* file = abs_node_id.file();
    // TODO: Consider asking our cpp_code_generator to map the location to a
    // debug location, in order to use Clang's rules for (eg) macro handling.
    auto loc = file->clang_source_locs().Get(abs_node_id.clang_source_loc_id());
    auto presumed_loc = file->cpp_file()->source_manager().getPresumedLoc(loc);
    return {
        .filename = presumed_loc.getFilename(),
        .line_number = static_cast<int32_t>(presumed_loc.getLine()),
        .column_number = static_cast<int32_t>(presumed_loc.getColumn()),
    };
  }

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
