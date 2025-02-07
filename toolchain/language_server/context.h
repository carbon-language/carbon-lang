// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_LANGUAGE_SERVER_CONTEXT_H_
#define CARBON_TOOLCHAIN_LANGUAGE_SERVER_CONTEXT_H_

#include <memory>
#include <string>

#include "common/map.h"
#include "toolchain/base/shared_value_stores.h"
#include "toolchain/diagnostics/diagnostic_consumer.h"
#include "toolchain/diagnostics/diagnostic_emitter.h"
#include "toolchain/diagnostics/file_diagnostics.h"
#include "toolchain/lex/tokenized_buffer.h"
#include "toolchain/parse/tree_and_subtrees.h"
#include "toolchain/sem_ir/file.h"
#include "toolchain/source/source_buffer.h"

namespace Carbon::LanguageServer {

// Context for LSP call handling.
class Context {
 public:
  // Cached information for an open file.
  class File {
   public:
    explicit File(std::string filename) : filename_(std::move(filename)) {}

    // Changes the file's text, updating dependent state.
    auto SetText(Context& context, llvm::StringRef text) -> void;

    auto tree_and_subtrees() const -> const Parse::TreeAndSubtrees& {
      return *tree_and_subtrees_;
    }

   private:
    // The filename, stable across instances.
    std::string filename_;

    // Current file content, and derived values.
    std::unique_ptr<SourceBuffer> source_;
    std::unique_ptr<SharedValueStores> value_stores_;
    std::unique_ptr<Lex::TokenizedBuffer> tokens_;
    std::unique_ptr<Parse::Tree> tree_;
    std::unique_ptr<Parse::TreeAndSubtrees> tree_and_subtrees_;
  };

  // `consumer` and `emitter` are required. `vlog_stream` is optional.
  explicit Context(llvm::raw_ostream* vlog_stream, DiagnosticConsumer* consumer)
      : vlog_stream_(vlog_stream),
        file_emitter_(consumer),
        no_loc_emitter_(consumer) {}

  // Returns a reference to the file if it's known, or diagnoses and returns
  // null.
  auto LookupFile(llvm::StringRef filename) -> File*;

  auto file_emitter() -> FileDiagnosticEmitter& { return file_emitter_; }
  auto no_loc_emitter() -> NoLocDiagnosticEmitter& { return no_loc_emitter_; }
  auto vlog_stream() -> llvm::raw_ostream* { return vlog_stream_; }

  auto files() -> Map<std::string, File>& { return files_; }

 private:
  // Diagnostic and output streams.
  llvm::raw_ostream* vlog_stream_;
  FileDiagnosticEmitter file_emitter_;
  NoLocDiagnosticEmitter no_loc_emitter_;

  // Content of files managed by the language client.
  Map<std::string, File> files_;
};

}  // namespace Carbon::LanguageServer

#endif  // CARBON_TOOLCHAIN_LANGUAGE_SERVER_CONTEXT_H_
