// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/language_server/context.h"

#include <memory>
#include <optional>
#include <utility>

#include "common/check.h"
#include "common/raw_string_ostream.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "llvm/TargetParser/Host.h"
#include "toolchain/base/clang_invocation.h"
#include "toolchain/base/shared_value_stores.h"
#include "toolchain/check/check.h"
#include "toolchain/diagnostics/consumer.h"
#include "toolchain/diagnostics/diagnostic.h"
#include "toolchain/diagnostics/emitter.h"
#include "toolchain/driver/compile_driver.h"
#include "toolchain/driver/compile_options.h"
#include "toolchain/lex/lex.h"
#include "toolchain/lex/tokenized_buffer.h"
#include "toolchain/parse/parse.h"
#include "toolchain/parse/tree_and_subtrees.h"

namespace Carbon::LanguageServer {

namespace {
// A consumer for turning diagnostics into a `textDocument/publishDiagnostics`
// notification.
// https://microsoft.github.io/language-server-protocol/specifications/lsp/3.17/specification/#textDocument_publishDiagnostics
class DiagnosticConsumer : public Diagnostics::Consumer {
 public:
  // Initializes params with the target file information.
  explicit DiagnosticConsumer(Context* context,
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
            .end = {.line = loc.line_number - 1,
                    .character = loc.column_number + loc.length - 1}};
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

// A virtual file corresponding to a file that has been opened and potentially
// edited by the language client.
class VirtualFile : public llvm::vfs::File {
 public:
  explicit VirtualFile(const Context::File* file) : file_(file) {}

  auto status() -> llvm::ErrorOr<llvm::vfs::Status> override {
    return llvm::vfs::Status(
        file_->filename(), llvm::sys::fs::UniqueID(0, 0),
        std::chrono::system_clock::now(), 0, 0, file_->text().size(),
        llvm::sys::fs::file_type::regular_file, llvm::sys::fs::all_all);
  }

  auto getBuffer(const llvm::Twine& /*name*/, int64_t /*file_size*/,
                 bool /*requires_null_terminator*/, bool /*is_volatile*/)
      -> llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> override {
    return llvm::MemoryBuffer::getMemBuffer(file_->text(), file_->filename());
  }

  auto close() -> std::error_code override { return std::error_code(); }

 private:
  const Context::File* file_;
};

// A virtual file system containing the documents whose contents were customized
// by the language client. We can't use InMemoryFileSystem for this, because it
// only supports setting the contents for each file once.
//
// TODO: Investigate whether we can replace this with clangd's `DraftStore`.
class VirtualFileSystem : public llvm::vfs::FileSystem {
 public:
  explicit VirtualFileSystem(Context* context) : context_(context) {}

  auto status(const llvm::Twine& path)
      -> llvm::ErrorOr<llvm::vfs::Status> override {
    std::string path_str = path.str();
    if (auto lookup_result = context_->files().Lookup(path_str)) {
      return llvm::vfs::Status(path_str, llvm::sys::fs::UniqueID(0, 0),
                               std::chrono::system_clock::now(), /*User=*/0,
                               /*Group=*/0, lookup_result.value().text().size(),
                               llvm::sys::fs::file_type::regular_file,
                               llvm::sys::fs::all_all);
    }
    return std::make_error_code(std::errc::no_such_file_or_directory);
  }

  auto openFileForRead(const llvm::Twine& path)
      -> llvm::ErrorOr<std::unique_ptr<llvm::vfs::File>> override {
    std::string path_str = path.str();
    if (auto lookup_result = context_->files().Lookup(path_str)) {
      return std::unique_ptr<llvm::vfs::File>(
          new VirtualFile(&lookup_result.value()));
    }
    return std::make_error_code(std::errc::no_such_file_or_directory);
  }

  auto dir_begin(const llvm::Twine& /*dir*/, std::error_code& ec)
      -> llvm::vfs::directory_iterator override {
    ec = std::make_error_code(std::errc::no_such_file_or_directory);
    return llvm::vfs::directory_iterator();
  }

  auto getCurrentWorkingDirectory() const
      -> llvm::ErrorOr<std::string> override {
    return std::string("");
  }

  auto setCurrentWorkingDirectory(const llvm::Twine& /*path*/)
      -> std::error_code override {
    return std::error_code();
  }

 private:
  Context* context_;
};

}  // namespace

Context::Context(const InstallPaths* installation,
                 llvm::raw_ostream* vlog_stream,
                 Diagnostics::Consumer* consumer,
                 clang::clangd::LSPBinder::RawOutgoing* outgoing)
    : installation_(installation),
      vlog_stream_(vlog_stream),
      file_emitter_(consumer),
      no_loc_emitter_(consumer),
      outgoing_(outgoing) {
  auto ls_fs = llvm::makeIntrusiveRefCnt<VirtualFileSystem>(this);
  auto vfs = llvm::makeIntrusiveRefCnt<llvm::vfs::OverlayFileSystem>(
      llvm::vfs::getRealFileSystem());
  vfs->pushOverlay(ls_fs);
  vfs_ = vfs;
}

auto Context::File::SetText(Context& context, std::optional<int64_t> version,
                            llvm::StringRef text) -> void {
  // Clear state dependent on the source text.
  compile_driver_.reset();

  text_ = text.str();

  // A consumer to gather diagnostics for the file.
  DiagnosticConsumer consumer(&context, uri_, version);

  // TODO: Make the processing asynchronous, to better handle rapid text
  // updates.

  llvm::raw_null_ostream null_stream;
  DriverEnv driver_env(context.vfs(), &context.installation(),
                       /*input_stream=*/nullptr, &null_stream, &null_stream,
                       /*fuzzing=*/false,
                       /*enable_leaking=*/false, &consumer);
  // TODO: Either use `raw_pwrite_stream` for all vlog streams or stop requiring
  // one in DriverEnv.
  driver_env.vlog_stream =
      static_cast<llvm::raw_pwrite_stream*>(context.vlog_stream());

  options_ = CompileOptions();
  options_.codegen_options->target = options_.codegen_options->host;
  options_.phase = CompileOptions::Phase::Check;
  options_.input_filenames.push_back(filename());
  options_.prelude_import = true;

  compile_driver_ = std::make_unique<CompileDriver>(&options_);
  auto map_input = [](llvm::StringRef) -> std::string { return ""; };
  if (!compile_driver_->Initialize(driver_env, map_input)) {
    context.PublishDiagnostics(consumer.params());
    return;
  }
  compile_driver_->Compile(driver_env);

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
