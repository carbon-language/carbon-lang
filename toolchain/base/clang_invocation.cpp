// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/base/clang_invocation.h"

#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/Utils.h"

namespace Carbon {

// The fake file name to use for the synthesized includes file.
static constexpr const char IncludesFileName[] = "<carbon Cpp imports>";

namespace {

// Used to convert diagnostics from the Clang driver to Carbon diagnostics.
class ClangDriverDiagnosticConsumer : public clang::DiagnosticConsumer {
 public:
  // Creates an instance with the location that triggers calling Clang.
  // `context` must not be null.
  explicit ClangDriverDiagnosticConsumer(Diagnostics::NoLocEmitter* emitter)
      : emitter_(emitter) {}

  // Generates a Carbon warning for each Clang warning and a Carbon error for
  // each Clang error or fatal.
  auto HandleDiagnostic(clang::DiagnosticsEngine::Level diag_level,
                        const clang::Diagnostic& info) -> void override {
    DiagnosticConsumer::HandleDiagnostic(diag_level, info);

    llvm::SmallString<256> message;
    info.FormatDiagnostic(message);

    switch (diag_level) {
      case clang::DiagnosticsEngine::Ignored:
      case clang::DiagnosticsEngine::Note:
      case clang::DiagnosticsEngine::Remark: {
        // TODO: Emit notes and remarks.
        break;
      }
      case clang::DiagnosticsEngine::Warning:
      case clang::DiagnosticsEngine::Error:
      case clang::DiagnosticsEngine::Fatal: {
        CARBON_DIAGNOSTIC(CppInteropDriverWarning, Warning, "{0}", std::string);
        CARBON_DIAGNOSTIC(CppInteropDriverError, Error, "{0}", std::string);
        emitter_->Emit(diag_level == clang::DiagnosticsEngine::Warning
                           ? CppInteropDriverWarning
                           : CppInteropDriverError,
                       message.str().str());
        break;
      }
    }
  }

 private:
  // Diagnostic emitter. Note that driver diagnostics don't have meaningful
  // locations attached.
  Diagnostics::NoLocEmitter* emitter_;
};

}  // namespace

static auto BuildClangInvocationImpl(
    Diagnostics::NoLocEmitter& emitter,
    llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> fs,
    llvm::ArrayRef<std::string> clang_path_and_args)
    -> std::unique_ptr<clang::CompilerInvocation> {
  ClangDriverDiagnosticConsumer diagnostics_consumer(&emitter);

  // The clang driver inconveniently wants an array of `const char*`, so convert
  // the arguments.
  llvm::SmallVector<const char*> driver_args(llvm::map_range(
      clang_path_and_args, [](const std::string& str) { return str.c_str(); }));

  // Add our include file name as the input file, and force it to be interpreted
  // as C++.
  driver_args.push_back("-x");
  driver_args.push_back("c++");
  driver_args.push_back(IncludesFileName);

  // Build a diagnostics engine. Note that we don't have any diagnostic options
  // yet; they're produced by running the driver.
  clang::DiagnosticOptions driver_diag_opts;
  llvm::IntrusiveRefCntPtr<clang::DiagnosticsEngine> driver_diags(
      clang::CompilerInstance::createDiagnostics(*fs, driver_diag_opts,
                                                 &diagnostics_consumer,
                                                 /*ShouldOwnClient=*/false));

  // Ask the driver to process the arguments and build a corresponding clang
  // frontend invocation.
  return clang::createInvocation(driver_args,
                                 {.Diags = driver_diags, .VFS = fs});
}

auto BuildClangInvocation(Diagnostics::Consumer& consumer,
                          llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> fs,
                          llvm::ArrayRef<std::string> clang_path_and_args)
    -> std::unique_ptr<clang::CompilerInvocation> {
  Diagnostics::ErrorTrackingConsumer error_tracker(consumer);
  Diagnostics::NoLocEmitter emitter(&error_tracker);

  // Forward to the implementation to avoid exposing `import_cpp` outside check.
  auto invocation = BuildClangInvocationImpl(emitter, fs, clang_path_and_args);

  // If Clang produced an error, throw away its invocation.
  if (error_tracker.seen_error()) {
    return nullptr;
  }

  return invocation;
}

}  // namespace Carbon
