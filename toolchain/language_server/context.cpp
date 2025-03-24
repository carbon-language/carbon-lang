// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/language_server/context.h"

#include <memory>

#include "common/check.h"
#include "common/raw_string_ostream.h"
#include "toolchain/base/shared_value_stores.h"
#include "toolchain/check/check.h"
#include "toolchain/diagnostics/diagnostic.h"
#include "toolchain/diagnostics/diagnostic_consumer.h"
#include "toolchain/lex/lex.h"
#include "toolchain/lex/tokenized_buffer.h"
#include "toolchain/parse/parse.h"
#include "toolchain/parse/tree_and_subtrees.h"

namespace Carbon::LanguageServer {

// A consumer for turning diagnostics into a `textDocument/publishDiagnostics`
// notification.
// https://microsoft.github.io/language-server-protocol/specifications/lsp/3.17/specification/#textDocument_publishDiagnostics
class PublishDiagnosticConsumer : public Diagnostics::Consumer {
 public:
  // Initializes params with the target file information.
  explicit PublishDiagnosticConsumer(Context* context,
                                     const clang::clangd::URIForFile& uri,
                                     std::optional<int64_t> version)
      : context_(context), params_{.uri = uri, .version = version} {}

  // Turns a diagnostic into an LSP diagnostic.
  auto HandleDiagnostic(Diagnostics::Diagnostic diagnostic) -> void override {
    const auto& message = diagnostic.messages[0];
    if (message.loc.filename != params_.uri.file()) {
      // `pushDiagnostic` requires diagnostics to be associated with a location
      // in the current file. Suppress diagnostics rooted in other files.
      // TODO: Consider if there's a better way to handle this.
      RawStringOstream stream;
      Diagnostics::StreamConsumer consumer(&stream);
      consumer.HandleDiagnostic(diagnostic);

      CARBON_DIAGNOSTIC(LanguageServerDiagnosticInWrongFile, Warning,
                        "dropping diagnostic in {0}:\n{1}", std::string,
                        std::string);
      context_->file_emitter().Emit(
          params_.uri.file(), LanguageServerDiagnosticInWrongFile,
          message.loc.filename.str(), stream.TakeStr());
      return;
    }

    // Add the main message.
    params_.diagnostics.push_back(clang::clangd::Diagnostic{
        .range = GetRange(message.loc),
        .severity = GetSeverity(diagnostic.level),
        .source = "carbon",
        .message = message.Format(),
    });
    // TODO: Figure out constructing URIs for note locations.
  }

  // Returns the constructed request.
  auto params() -> const clang::clangd::PublishDiagnosticsParams& {
    return params_;
  }

 private:
  // Returns the LSP range for a diagnostic. Note that Carbon uses 1-based
  // numbers while LSP uses 0-based.
  auto GetRange(const Diagnostics::Loc& loc) -> clang::clangd::Range {
    return {.start = {.line = loc.line_number - 1,
                      .character = loc.column_number - 1},
            .end = {.line = loc.line_number,
                    .character = loc.column_number + loc.length}};
  }

  // Converts a diagnostic level to an LSP severity.
  auto GetSeverity(Diagnostics::Level level) -> int {
    // https://microsoft.github.io/language-server-protocol/specifications/lsp/3.17/specification/#diagnosticSeverity
    enum class DiagnosticSeverity {
      Error = 1,
      Warning = 2,
      Information = 3,
      Hint = 4,
    };

    switch (level) {
      case Diagnostics::Level::Error:
        return static_cast<int>(DiagnosticSeverity::Error);
      case Diagnostics::Level::Warning:
        return static_cast<int>(DiagnosticSeverity::Warning);
      default:
        CARBON_FATAL("Unexpected diagnostic level: {0}", level);
    }
  }

  Context* context_;
  clang::clangd::PublishDiagnosticsParams params_;
};

auto Context::File::SetText(Context& context, std::optional<int64_t> version,
                            llvm::StringRef text) -> void {
  // Clear state dependent on the source text.
  tree_and_subtrees_.reset();
  tree_.reset();
  tokens_.reset();
  value_stores_.reset();
  source_.reset();

  // A consumer to gather diagnostics for the file.
  PublishDiagnosticConsumer consumer(&context, uri_, version);

  // TODO: Make the processing asynchronous, to better handle rapid text
  // updates.
  CARBON_CHECK(!source_ && !value_stores_ && !tokens_ && !tree_,
               "We currently cache everything together");
  // TODO: Diagnostics should be passed to the LSP instead of dropped.
  std::optional source =
      SourceBuffer::MakeFromStringCopy(uri_.file(), text, consumer);
  if (!source) {
    // Failing here should be rare, but provide stub data for recovery so that
    // we can have a simple API.
    source = SourceBuffer::MakeFromStringCopy(uri_.file(), "", consumer);
    CARBON_CHECK(source, "Making an empty buffer should always succeed");
  }
  source_ = std::make_unique<SourceBuffer>(std::move(*source));
  value_stores_ = std::make_unique<SharedValueStores>();
  tokens_ = std::make_unique<Lex::TokenizedBuffer>(
      Lex::Lex(*value_stores_, *source_, consumer));
  tree_ = std::make_unique<Parse::Tree>(
      Parse::Parse(*tokens_, consumer, context.vlog_stream()));
  tree_and_subtrees_ =
      std::make_unique<Parse::TreeAndSubtrees>(*tokens_, *tree_);

  SemIR::File sem_ir(tree_.get(), SemIR::CheckIRId(0), tree_->packaging_decl(),
                     *value_stores_, uri_.file().str());
  auto getter = [this]() -> const Parse::TreeAndSubtrees& {
    return *tree_and_subtrees_;
  };
  // TODO: Support cross-file checking when multiple files have edits.
  llvm::SmallVector<Check::Unit> units = {{.consumer = &consumer,
                                           .value_stores = value_stores_.get(),
                                           .timings = nullptr,
                                           .tree_and_subtrees_getter = getter,
                                           .sem_ir = &sem_ir}};
  llvm::IntrusiveRefCntPtr<llvm::vfs::InMemoryFileSystem> fs =
      new llvm::vfs::InMemoryFileSystem;
  // TODO: Include the prelude.
  Check::CheckParseTrees(units, /*prelude_import=*/false, fs,
                         context.vlog_stream(), /*fuzzing=*/false);

  // Note we need to publish diagnostics even when empty.
  // TODO: Consider caching previously published diagnostics and only publishing
  // when they change.
  context.PublishDiagnostics(consumer.params());
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
