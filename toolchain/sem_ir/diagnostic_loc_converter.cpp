// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/sem_ir/diagnostic_loc_converter.h"

#include "clang/Frontend/DiagnosticRenderer.h"

namespace Carbon::SemIR {

static auto ConvertPresumedLocToDiagnosticsLoc(clang::FullSourceLoc loc,
                                               clang::PresumedLoc presumed_loc)
    -> Diagnostics::Loc {
  llvm::StringRef line;

  // Ask the Clang SourceManager for the contents of the line containing this
  // location.
  // TODO: If this location is in our generated header, use the source text from
  // the presumed location (the Carbon source file) as the snippet instead.
  bool loc_invalid = false;
  const auto& src_mgr = loc.getManager();
  auto [file_id, offset] = src_mgr.getDecomposedSpellingLoc(loc);
  auto loc_line = src_mgr.getLineNumber(file_id, offset, &loc_invalid);
  if (!loc_invalid) {
    auto start_of_line = src_mgr.translateLineCol(file_id, loc_line, 1);
    line = src_mgr.getCharacterData(start_of_line, &loc_invalid);
    line = line.take_until([](char c) { return c == '\n'; });
  }

  return {.filename = presumed_loc.getFilename(),
          .line = loc_invalid ? "" : line,
          .line_number = static_cast<int32_t>(presumed_loc.getLine()),
          .column_number = static_cast<int32_t>(presumed_loc.getColumn()),
          .length = loc_invalid ? -1 : 1};
}

namespace {
// A diagnostics "renderer" that renders the diagnostic into an array of
// importing contexts based on the C++ include stack.
class ClangImportCollector : public clang::DiagnosticRenderer {
 public:
  explicit ClangImportCollector(
      const clang::LangOptions& lang_opts, clang::DiagnosticOptions& diag_opts,
      llvm::SmallVectorImpl<DiagnosticLocConverter::ImportLoc>* imports)
      : DiagnosticRenderer(lang_opts, diag_opts), imports_(imports) {}

  void emitDiagnosticMessage(clang::FullSourceLoc loc, clang::PresumedLoc ploc,
                             clang::DiagnosticsEngine::Level /*level*/,
                             llvm::StringRef message,
                             llvm::ArrayRef<clang::CharSourceRange> /*ranges*/,
                             clang::DiagOrStoredDiag /*info*/) override {
    if (!emitted_message_) {
      emitted_message_ = true;
      return;
    }
    // This is an "in macro expanded here" diagnostic that Clang emits after the
    // emitted diagnostic. We treat that as another form of context location.
    imports_->push_back(
        {.loc = ConvertPresumedLocToDiagnosticsLoc(loc, ploc),
         .kind = DiagnosticLocConverter::ImportLoc::CppMacroExpansion,
         .imported_name = message});
  }

  void emitDiagnosticLoc(
      clang::FullSourceLoc /*loc*/, clang::PresumedLoc /*ploc*/,
      clang::DiagnosticsEngine::Level /*level*/,
      llvm::ArrayRef<clang::CharSourceRange> /*ranges*/) override {}
  void emitCodeContext(
      clang::FullSourceLoc /*loc*/, clang::DiagnosticsEngine::Level /*level*/,
      llvm::SmallVectorImpl<clang::CharSourceRange>& /*ranges*/,
      llvm::ArrayRef<clang::FixItHint> /*hints*/) override {}

  void emitIncludeLocation(clang::FullSourceLoc loc,
                           clang::PresumedLoc ploc) override {
    // TODO: If this location is for a `#include` in the generated C++ includes
    // buffer that corresponds to a carbon import, report it as being an Import
    // instead of a CppInclude.
    imports_->push_back(
        {.loc = ConvertPresumedLocToDiagnosticsLoc(loc, ploc),
         .kind = DiagnosticLocConverter::ImportLoc::CppInclude});
  }
  void emitImportLocation(clang::FullSourceLoc loc, clang::PresumedLoc ploc,
                          llvm::StringRef module_name) override {
    imports_->push_back(
        {.loc = ConvertPresumedLocToDiagnosticsLoc(loc, ploc),
         .kind = DiagnosticLocConverter::ImportLoc::CppModuleImport,
         .imported_name = module_name});
  }
  void emitBuildingModuleLocation(clang::FullSourceLoc loc,
                                  clang::PresumedLoc ploc,
                                  llvm::StringRef module_name) override {
    imports_->push_back(
        {.loc = ConvertPresumedLocToDiagnosticsLoc(loc, ploc),
         .kind = DiagnosticLocConverter::ImportLoc::CppModuleImport,
         .imported_name = module_name});
  }

 private:
  llvm::SmallVectorImpl<DiagnosticLocConverter::ImportLoc>* imports_;
  // Whether we've emitted the primary diagnostic message or not. Any diagnostic
  // emitted after this is an "in macro expansion" note that we want to capture
  // as context.
  bool emitted_message_ = false;
};
}  // namespace

auto DiagnosticLocConverter::ConvertWithImports(LocId loc_id,
                                                bool token_only) const
    -> LocAndImports {
  llvm::SmallVector<SemIR::AbsoluteNodeId> absolute_node_ids =
      SemIR::GetAbsoluteNodeId(sem_ir_, loc_id);
  auto final_node_id = absolute_node_ids.pop_back_val();

  // Convert the final location.
  LocAndImports result = {.loc = ConvertImpl(final_node_id, token_only)};

  // Convert the import locations.
  for (const auto& absolute_node_id : absolute_node_ids) {
    if (!absolute_node_id.node_id().has_value()) {
      // TODO: Add an `ImportLoc` pointing at the prelude for the case where
      // we don't have a location.
      continue;
    }
    result.imports.push_back({.loc = ConvertImpl(absolute_node_id, false).loc});
  }

  // Convert the C++ import locations.
  if (final_node_id.check_ir_id() == SemIR::CheckIRId::Cpp) {
    const clang::ASTUnit* ast = sem_ir_->clang_ast_unit();
    // Collect the location backtrace that Clang would use for an error here.
    ClangImportCollector(ast->getLangOpts(),
                         ast->getDiagnostics().getDiagnosticOptions(),
                         &result.imports)
        .emitDiagnostic(
            clang::FullSourceLoc(sem_ir_->clang_source_locs().Get(
                                     final_node_id.clang_source_loc_id()),
                                 ast->getSourceManager()),
            clang::DiagnosticsEngine::Error, "", {}, {});
  }

  return result;
}

auto DiagnosticLocConverter::Convert(LocId loc_id, bool token_only) const
    -> Diagnostics::ConvertedLoc {
  llvm::SmallVector<SemIR::AbsoluteNodeId> absolute_node_ids =
      SemIR::GetAbsoluteNodeId(sem_ir_, loc_id);
  return ConvertImpl(absolute_node_ids.back(), token_only);
}

auto DiagnosticLocConverter::ConvertImpl(SemIR::AbsoluteNodeId absolute_node_id,
                                         bool token_only) const
    -> Diagnostics::ConvertedLoc {
  if (absolute_node_id.check_ir_id() == SemIR::CheckIRId::Cpp) {
    return ConvertImpl(absolute_node_id.clang_source_loc_id());
  }

  return ConvertImpl(absolute_node_id.check_ir_id(), absolute_node_id.node_id(),
                     token_only);
}

auto DiagnosticLocConverter::ConvertImpl(SemIR::CheckIRId check_ir_id,
                                         Parse::NodeId node_id,
                                         bool token_only) const
    -> Diagnostics::ConvertedLoc {
  CARBON_CHECK(check_ir_id != SemIR::CheckIRId::Cpp);
  const auto& tree_and_subtrees =
      tree_and_subtrees_getters_->Get(check_ir_id)();
  return tree_and_subtrees.NodeToDiagnosticLoc(node_id, token_only);
}

auto DiagnosticLocConverter::ConvertImpl(
    ClangSourceLocId clang_source_loc_id) const -> Diagnostics::ConvertedLoc {
  clang::SourceLocation clang_loc =
      sem_ir_->clang_source_locs().Get(clang_source_loc_id);

  CARBON_CHECK(sem_ir_->clang_ast_unit());
  const auto& src_mgr = sem_ir_->clang_ast_unit()->getSourceManager();
  clang::PresumedLoc presumed_loc = src_mgr.getPresumedLoc(clang_loc);
  if (presumed_loc.isInvalid()) {
    return Diagnostics::ConvertedLoc();
  }
  unsigned offset = src_mgr.getDecomposedLoc(clang_loc).second;

  return Diagnostics::ConvertedLoc{
      .loc = ConvertPresumedLocToDiagnosticsLoc(
          clang::FullSourceLoc(clang_loc, src_mgr), presumed_loc),
      .last_byte_offset = static_cast<int32_t>(offset)};
}

}  // namespace Carbon::SemIR
