// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_LANGUAGE_SERVER_CONTEXT_H_
#define CARBON_TOOLCHAIN_LANGUAGE_SERVER_CONTEXT_H_

#include <memory>
#include <string>

#include "clang-tools-extra/clangd/LSPBinder.h"
#include "common/map.h"
#include "toolchain/base/install_paths.h"
#include "toolchain/base/shared_value_stores.h"
#include "toolchain/diagnostics/consumer.h"
#include "toolchain/diagnostics/emitter.h"
#include "toolchain/diagnostics/file_diagnostics.h"
#include "toolchain/driver/compile_driver.h"
#include "toolchain/lex/tokenized_buffer.h"
#include "toolchain/parse/tree_and_subtrees.h"
#include "toolchain/sem_ir/file.h"

namespace Carbon::LanguageServer {

// Context for LSP call handling.
class Context {
 public:
  // Cached information for an open file.
  class File {
   public:
    explicit File(clang::clangd::URIForFile uri)
        : uri_(std::move(uri)), filename_(uri_.file().str()) {}

    // Changes the file's text, updating dependent state.
    auto SetText(Context& context, std::optional<int64_t> version,
                 llvm::StringRef text) -> void;

    auto filename() const -> llvm::StringRef { return filename_; }
    auto text() const -> llvm::StringRef { return text_; }

    auto tree_and_subtrees() const -> const Parse::TreeAndSubtrees& {
      CARBON_CHECK(compile_driver_);
      return compile_driver_->units()[compile_driver_->first_input_index()]
          ->parse_tree_and_subtrees();
    }

   private:
    // The filename, stable across instances.
    clang::clangd::URIForFile uri_;
    std::string filename_;

    // Current file content, and derived values.
    std::string text_;
    CompileOptions options_;
    std::unique_ptr<CompileDriver> compile_driver_;
  };

  // `vlog_stream` is optional; other parameters are required.
  explicit Context(const InstallPaths* installation,
                   llvm::raw_ostream* vlog_stream,
                   Diagnostics::Consumer* consumer,
                   clang::clangd::LSPBinder::RawOutgoing* outgoing);

  // Returns the virtual filesystem.
  auto vfs() -> llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem>& {
    return vfs_;
  }

  // Returns a reference to the file if it's known, or diagnoses and returns
  // null.
  auto LookupFile(llvm::StringRef filename) -> File*;

  // Wrapper for LSP notification.
  auto PublishDiagnostics(clang::clangd::PublishDiagnosticsParams params)
      -> void {
    outgoing_->notify("textDocument/publishDiagnostics", params);
  }

  auto installation() -> const InstallPaths& { return *installation_; }

  auto vlog_stream() -> llvm::raw_ostream* { return vlog_stream_; }
  auto file_emitter() -> Diagnostics::FileEmitter& { return file_emitter_; }
  auto no_loc_emitter() -> Diagnostics::NoLocEmitter& {
    return no_loc_emitter_;
  }

  auto files() -> Map<std::string, File>& { return files_; }

 private:
  const InstallPaths* installation_;

  // Diagnostic and output streams.
  llvm::raw_ostream* vlog_stream_;
  Diagnostics::FileEmitter file_emitter_;
  Diagnostics::NoLocEmitter no_loc_emitter_;
  clang::clangd::LSPBinder::RawOutgoing* outgoing_;

  // Shared virtual filesystem.
  llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> vfs_;

  // Content of files managed by the language client.
  Map<std::string, File> files_;
};

}  // namespace Carbon::LanguageServer

#endif  // CARBON_TOOLCHAIN_LANGUAGE_SERVER_CONTEXT_H_
