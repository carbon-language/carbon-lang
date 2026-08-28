// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <string>
#include <utility>
#include <vector>

#include "common/raw_string_ostream.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "toolchain/format/format.h"
#include "toolchain/language_server/handle.h"

namespace Carbon::LanguageServer {

// Returns the range covering the entire document text.
static auto GetFullDocumentRange(llvm::StringRef text) -> clang::clangd::Range {
  int line_count = llvm::count(text, '\n') + 1;
  return clang::clangd::Range{
      .start = {.line = 0, .character = 0},
      .end = {.line = line_count, .character = 0},
  };
}

auto HandleFormatting(
    Context& context, const clang::clangd::DocumentFormattingParams& params,
    llvm::function_ref<
        auto(llvm::Expected<std::vector<clang::clangd::TextEdit>>)->void>
        on_done) -> void {
  auto* file = context.LookupFile(params.textDocument.uri.file());
  if (!file) {
    return;
  }

  RawStringOstream out;
  if (!Format::Format(file->tokens(), out)) {
    out.clear();
    on_done(std::vector<clang::clangd::TextEdit>());
    return;
  }

  std::string formatted_text = out.TakeStr();
  if (formatted_text == file->text()) {
    on_done(std::vector<clang::clangd::TextEdit>());
    return;
  }

  std::vector<clang::clangd::TextEdit> edits;
  edits.push_back(clang::clangd::TextEdit{
      .range = GetFullDocumentRange(file->text()),
      .newText = std::move(formatted_text),
  });
  on_done(std::move(edits));
}

}  // namespace Carbon::LanguageServer
