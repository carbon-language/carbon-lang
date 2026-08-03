// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/sem_ir/diagnostic_loc_converter.h"

#include <algorithm>

#include "clang/Basic/SourceManager.h"
#include "clang/Frontend/DiagnosticRenderer.h"
#include "common/check.h"

namespace Carbon::SemIR {

// Returns the bytes `end` reaches past `loc` on the line `loc` is on, or 1 when
// it says nothing useful about an extent. Bytes rather than columns because
// that is what `Diagnostics::Loc::length` holds; rendering converts.
//
// `end` must name the byte past the range, which is what a character range
// gives. A token range's end names the start of its last token instead, so
// measuring one here would drop that token from the underline; run it through
// `clang::Lexer::getAsCharRange` first.
//
// A range that runs off the end of the line is clamped there, the way a
// location spanning several lines is: a snippet shows one line, so an
// underline that left it would have nowhere to go.
static auto MeasureRange(clang::FullSourceLoc loc, clang::SourceLocation end,
                         llvm::StringRef line, int32_t column) -> int32_t {
  if (end.isInvalid()) {
    return 1;
  }
  const auto& src_mgr = loc.getManager();
  auto [file_id, offset] = src_mgr.getDecomposedSpellingLoc(loc);
  auto [end_file_id, end_offset] = src_mgr.getDecomposedSpellingLoc(end);
  if (end_file_id != file_id || end_offset <= offset) {
    return 1;
  }
  auto length = static_cast<int32_t>(end_offset - offset);
  auto to_end_of_line = static_cast<int32_t>(line.size()) - (column - 1);
  return std::max(1, std::min(length, to_end_of_line));
}

// Returns the Carbon location for a Clang one, underlining `range` where it
// says how far the location reaches.
static auto ConvertPresumedLocToDiagnosticsLoc(clang::FullSourceLoc loc,
                                               clang::PresumedLoc presumed_loc,
                                               clang::CharSourceRange range)
    -> Diagnostics::Loc;

auto ConvertClangRangeToLoc(const clang::SourceManager& src_mgr,
                            clang::CharSourceRange range) -> Diagnostics::Loc {
  CARBON_CHECK(range.isCharRange() || range.getBegin() == range.getEnd(),
               "A token range's end names the start of its last token rather "
               "than an extent; widen it with `Lexer::getAsCharRange` first.");
  clang::SourceLocation begin = range.getBegin();
  if (begin.isInvalid()) {
    return Diagnostics::Loc();
  }
  clang::PresumedLoc presumed_loc = src_mgr.getPresumedLoc(begin);
  if (presumed_loc.isInvalid()) {
    return Diagnostics::Loc();
  }
  return ConvertPresumedLocToDiagnosticsLoc(
      clang::FullSourceLoc(begin, src_mgr), presumed_loc, range);
}

static auto ConvertPresumedLocToDiagnosticsLoc(clang::FullSourceLoc loc,
                                               clang::PresumedLoc presumed_loc,
                                               clang::CharSourceRange range)
    -> Diagnostics::Loc {
  llvm::StringRef line;
  llvm::StringRef file_text;

  // Ask the Clang SourceManager for the contents of the line containing this
  // location, and for the file it is a slice of, which is what lets a snippet
  // show the lines between two spans that are close together.
  // TODO: If this location is in our generated header, use the source text from
  // the presumed location (the Carbon source file) as the snippet instead.
  // TODO: A `#line` directive skews the presumed line numbers here against the
  // physical lines of `file_text`, so the lines a snippet shows between two
  // nearby spans can be the wrong ones. Cosmetic and bounded, but fixing it
  // means carrying spelling line numbers alongside the presumed ones.
  bool loc_invalid = false;
  const auto& src_mgr = loc.getManager();
  auto [file_id, offset] = src_mgr.getDecomposedSpellingLoc(loc);
  auto loc_line = src_mgr.getLineNumber(file_id, offset, &loc_invalid);
  if (!loc_invalid) {
    file_text = src_mgr.getBufferData(file_id, &loc_invalid);
  }
  if (!loc_invalid) {
    auto start_of_line = src_mgr.translateLineCol(file_id, loc_line, 1);
    line = src_mgr.getCharacterData(start_of_line, &loc_invalid);
    line = line.take_until([](char c) { return c == '\n'; });
  }

  auto column = static_cast<int32_t>(presumed_loc.getColumn());
  return {.filename = presumed_loc.getFilename(),
          .line = loc_invalid ? "" : line,
          .file_text = loc_invalid ? "" : file_text,
          .line_number = static_cast<int32_t>(presumed_loc.getLine()),
          .column_number = column,
          .length = loc_invalid
                        ? -1
                        : MeasureRange(loc, range.getEnd(), line, column)};
}

namespace {
// A diagnostics "renderer" that renders the diagnostic into an array of
// importing contexts based on the C++ include stack.
class ClangImportCollector : public clang::DiagnosticRenderer {
 public:
  explicit ClangImportCollector(
      const clang::LangOptions& lang_opts,
      const clang::DiagnosticOptions& diag_opts,
      llvm::SmallVectorImpl<DiagnosticLocConverter::ImportLoc>* imports)
      : DiagnosticRenderer(lang_opts,
                           // Work around lack of const-correctness in Clang.
                           const_cast<clang::DiagnosticOptions&>(diag_opts)),
        imports_(imports) {}

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
        {.loc = ConvertPresumedLocToDiagnosticsLoc(loc, ploc,
                                                   clang::CharSourceRange()),
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
        {.loc = ConvertPresumedLocToDiagnosticsLoc(loc, ploc,
                                                   clang::CharSourceRange()),
         .kind = DiagnosticLocConverter::ImportLoc::CppInclude});
  }
  void emitImportLocation(clang::FullSourceLoc loc, clang::PresumedLoc ploc,
                          llvm::StringRef module_name) override {
    imports_->push_back(
        {.loc = ConvertPresumedLocToDiagnosticsLoc(loc, ploc,
                                                   clang::CharSourceRange()),
         .kind = DiagnosticLocConverter::ImportLoc::CppModuleImport,
         .imported_name = module_name});
  }
  void emitBuildingModuleLocation(clang::FullSourceLoc loc,
                                  clang::PresumedLoc ploc,
                                  llvm::StringRef module_name) override {
    imports_->push_back(
        {.loc = ConvertPresumedLocToDiagnosticsLoc(loc, ploc,
                                                   clang::CharSourceRange()),
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
  llvm::SmallVector<AbsoluteNodeRef> absolute_node_refs =
      GetAbsoluteNodeRef(sem_ir_, loc_id);
  auto final_node = absolute_node_refs.pop_back_val();

  // Convert the final location.
  LocAndImports result = {.loc = ConvertImpl(final_node, token_only)};

  // Convert the import locations.
  for (const auto& absolute_node_ref : absolute_node_refs) {
    if (!absolute_node_ref.node_id().has_value()) {
      // TODO: Add an `ImportLoc` pointing at the prelude for the case where
      // we don't have a location.
      continue;
    }
    result.imports.push_back(
        {.loc = ConvertImpl(absolute_node_ref, false).loc});
  }

  // Convert the C++ import locations.
  if (final_node.is_cpp()) {
    const File* file = final_node.file();
    CARBON_CHECK(file->cpp_file(),
                 "Converting C++ location before C++ file is set");

    // Collect the location backtrace that Clang would use for an error here.
    ClangImportCollector(file->cpp_file()->lang_options(),
                         file->cpp_file()->diagnostic_options(),
                         &result.imports)
        .emitDiagnostic(
            clang::FullSourceLoc(file->clang_source_locs()
                                     .Get(final_node.clang_source_loc_id())
                                     .getBegin(),
                                 file->cpp_file()->source_manager()),
            clang::DiagnosticsEngine::Error, "", {}, {});
  }

  return result;
}

auto DiagnosticLocConverter::Convert(LocId loc_id, bool token_only) const
    -> Diagnostics::ConvertedLoc {
  llvm::SmallVector<AbsoluteNodeRef> absolute_node_refs =
      GetAbsoluteNodeRef(sem_ir_, loc_id);
  return ConvertImpl(absolute_node_refs.back(), token_only);
}

auto DiagnosticLocConverter::ConvertImpl(AbsoluteNodeRef absolute_node_ref,
                                         bool token_only) const
    -> Diagnostics::ConvertedLoc {
  if (absolute_node_ref.is_cpp()) {
    return ConvertImpl(absolute_node_ref.file(),
                       absolute_node_ref.clang_source_loc_id());
  }

  return ConvertImpl(absolute_node_ref.check_ir_id(),
                     absolute_node_ref.node_id(), token_only);
}

auto DiagnosticLocConverter::ConvertImpl(CheckIRId check_ir_id,
                                         Parse::NodeId node_id,
                                         bool token_only) const
    -> Diagnostics::ConvertedLoc {
  const auto& tree_and_subtrees =
      tree_and_subtrees_getters_->Get(check_ir_id)();
  return tree_and_subtrees.NodeToDiagnosticLoc(node_id, token_only);
}

auto DiagnosticLocConverter::ConvertImpl(
    const File* file, ClangSourceLocId clang_source_loc_id) const
    -> Diagnostics::ConvertedLoc {
  clang::CharSourceRange clang_range = clang::CharSourceRange::getCharRange(
      file->clang_source_locs().Get(clang_source_loc_id));
  clang::SourceLocation clang_loc = clang_range.getBegin();

  CARBON_CHECK(file->cpp_file());
  const auto& src_mgr = file->cpp_file()->source_manager();
  if (clang_loc.isInvalid() || src_mgr.getPresumedLoc(clang_loc).isInvalid()) {
    return Diagnostics::ConvertedLoc();
  }
  unsigned offset = src_mgr.getDecomposedLoc(clang_loc).second;

  return Diagnostics::ConvertedLoc{
      .loc = ConvertClangRangeToLoc(src_mgr, clang_range),
      .last_byte_offset = static_cast<int32_t>(offset)};
}

}  // namespace Carbon::SemIR
