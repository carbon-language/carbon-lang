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
#include "toolchain/driver/codegen_options.h"
#include "toolchain/driver/compile_driver.h"
#include "toolchain/driver/compile_options.h"
#include "toolchain/language_server/sem_ir_index.h"
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
        : uri_(std::move(uri)),
          filename_(uri_.file().str()),
          options_(&codegen_options_) {}

    // Changes the file's text, updating dependent state.
    auto SetText(Context& context, std::optional<int64_t> version,
                 llvm::StringRef text) -> void;

    auto uri() const -> const clang::clangd::URIForFile& { return uri_; }
    auto filename() const -> llvm::StringRef { return filename_; }
    auto text() const -> llvm::StringRef { return text_; }

    auto tree_and_subtrees() const -> const Parse::TreeAndSubtrees& {
      return unit().parse_tree_and_subtrees();
    }

    auto tokens() const -> const Lex::TokenizedBuffer& {
      return unit().tokens();
    }

    // Returns the checked IR, or null if checking didn't get far enough to
    // produce one.
    auto sem_ir() const -> const SemIR::File* {
      const auto& compilation_unit = unit();
      return compilation_unit.has_sem_ir() ? &compilation_unit.sem_ir()
                                           : nullptr;
    }

    // Returns an index of this file's instructions by token, building it if
    // this is the first query since the text last changed. Returns null if
    // there's no checked IR to index.
    //
    // This is deliberately not built by `SetText`: most text changes are
    // followed by another text change rather than by a query, and the work
    // would land on the path that produces diagnostics, which is the latency
    // users actually notice.
    auto sem_ir_index() const -> const SemIRIndex*;

   private:
    auto unit() const -> const CompilationUnit& {
      CARBON_CHECK(compile_driver_);
      return *compile_driver_->units()[compile_driver_->first_input_index()];
    }

    // The filename, stable across instances.
    clang::clangd::URIForFile uri_;
    std::string filename_;

    // Current file content, and derived values.
    std::string text_;

    CodegenOptions codegen_options_;
    CompileOptions options_;
    std::unique_ptr<CompileDriver> compile_driver_;

    // Built on demand by `sem_ir_index()`, and discarded by `SetText`.
    mutable std::optional<SemIRIndex> sem_ir_index_;
  };

  // `vlog_stream` is optional; other parameters are required.
  explicit Context(const InstallPaths* installation,
                   llvm::raw_ostream* vlog_stream,
                   Diagnostics::Consumer* consumer,
                   clang::clangd::LSPBinder::RawOutgoing* outgoing,
                   bool prelude_import);

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

  auto prelude_import() const -> bool { return prelude_import_; }

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

  bool prelude_import_;
};

}  // namespace Carbon::LanguageServer

#endif  // CARBON_TOOLCHAIN_LANGUAGE_SERVER_CONTEXT_H_
