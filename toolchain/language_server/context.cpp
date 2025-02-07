// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/language_server/context.h"

#include <memory>

#include "toolchain/base/shared_value_stores.h"
#include "toolchain/diagnostics/null_diagnostics.h"
#include "toolchain/lex/lex.h"
#include "toolchain/lex/tokenized_buffer.h"
#include "toolchain/parse/parse.h"
#include "toolchain/parse/tree_and_subtrees.h"

namespace Carbon::LanguageServer {

auto Context::File::SetText(Context& context, llvm::StringRef text) -> void {
  // Clear state dependent on the source text.
  tree_and_subtrees_.reset();
  tree_.reset();
  tokens_.reset();
  value_stores_.reset();
  source_.reset();

  // TODO: Make the processing asynchronous, to better handle rapid text
  // updates.
  CARBON_CHECK(!source_ && !value_stores_ && !tokens_ && !tree_,
               "We currently cache everything together");
  // TODO: Diagnostics should be passed to the LSP instead of dropped.
  auto& null_consumer = NullDiagnosticConsumer();
  std::optional source =
      SourceBuffer::MakeFromStringCopy(filename_, text, null_consumer);
  if (!source) {
    // Failing here should be rare, but provide stub data for recovery so that
    // we can have a simple API.
    source = SourceBuffer::MakeFromStringCopy(filename_, "", null_consumer);
    CARBON_CHECK(source, "Making an empty buffer should always succeed");
  }
  source_ = std::make_unique<SourceBuffer>(std::move(*source));
  value_stores_ = std::make_unique<SharedValueStores>();
  tokens_ = std::make_unique<Lex::TokenizedBuffer>(
      Lex::Lex(*value_stores_, *source_, null_consumer));
  tree_ = std::make_unique<Parse::Tree>(
      Parse::Parse(*tokens_, null_consumer, context.vlog_stream()));
  tree_and_subtrees_ =
      std::make_unique<Parse::TreeAndSubtrees>(*tokens_, *tree_);
}

auto Context::LookupFile(llvm::StringRef filename) -> File* {
  if (!filename.ends_with(".carbon")) {
    CARBON_DIAGNOSTIC(LanguageServerFileUnsupported, Warning,
                      "non-Carbon file requested");
    file_emitter_.Emit(filename, LanguageServerFileUnsupported);
    return nullptr;
  }

  if (auto lookup_result = files().Lookup(filename)) {
    return &lookup_result.value();
  } else {
    CARBON_DIAGNOSTIC(LanguageServerFileUnknown, Warning,
                      "unknown file requested");
    file_emitter_.Emit(filename, LanguageServerFileUnknown);
    return nullptr;
  }
}

}  // namespace Carbon::LanguageServer
