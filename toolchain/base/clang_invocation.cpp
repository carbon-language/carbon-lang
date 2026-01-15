// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/base/clang_invocation.h"

#include <filesystem>
#include <string>

#include "clang/Driver/CreateInvocationFromArgs.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/Utils.h"
#include "common/string_helpers.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FormatVariadic.h"

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

auto BuildClangInvocation(Diagnostics::Consumer& consumer,
                          llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> fs,
                          const InstallPaths& install_paths,
                          llvm::StringRef target_str,
                          llvm::ArrayRef<llvm::StringRef> extra_args)
    -> std::unique_ptr<clang::CompilerInvocation> {
  Diagnostics::ErrorTrackingConsumer error_tracker(consumer);
  Diagnostics::NoLocEmitter emitter(&error_tracker);

  ClangDriverDiagnosticConsumer diagnostics_consumer(&emitter);

  llvm::SmallVector<std::string> args;
  args.push_back("--start-no-unused-arguments");
  AppendDefaultClangArgs(install_paths, target_str, args);
  args.push_back("--end-no-unused-arguments");
  args.append({
      llvm::formatv("--target={0}", target_str).str(),

      // Add our include file name as the input file, and force it to be
      // interpreted as C++.
      "-x",
      "c++",
      IncludesFileName,
  });

  // The clang driver inconveniently wants an array of `const char*`, so convert
  // the arguments.
  llvm::BumpPtrAllocator alloc;
  llvm::SmallVector<const char*> cstr_args = BuildCStrArgs(
      install_paths.clang_path().native(), args, extra_args, alloc);

  // Build a diagnostics engine. Note that we don't have any diagnostic options
  // yet; they're produced by running the driver.
  clang::DiagnosticOptions driver_diag_opts;
  llvm::IntrusiveRefCntPtr<clang::DiagnosticsEngine> driver_diags(
      clang::CompilerInstance::createDiagnostics(*fs, driver_diag_opts,
                                                 &diagnostics_consumer,
                                                 /*ShouldOwnClient=*/false));

  // Ask the driver to process the arguments and build a corresponding clang
  // frontend invocation.
  auto invocation =
      clang::createInvocation(cstr_args, {.Diags = driver_diags, .VFS = fs});

  // If Clang produced an error, throw away its invocation.
  if (error_tracker.seen_error()) {
    return nullptr;
  }

  if (invocation) {
    // Do not emit Clang's name and version as the creator of the output file.
    invocation->getCodeGenOpts().EmitVersionIdentMetadata = false;
    invocation->getCodeGenOpts().DiscardValueNames = false;
  }

  return invocation;
}

auto AppendDefaultClangArgs(const InstallPaths& install_paths,
                            llvm::StringRef target_str,
                            llvm::SmallVectorImpl<std::string>& args) -> void {
  args.append({
      // Enable PIE by default, but allow it to be overridden by Clang
      // arguments. Clang's default is configurable, but we'd like our
      // defaults to be more stable.
      // TODO: Decide if we want this.
      "-fPIE",

      // Override runtime library defaults.
      //
      // TODO: We should consider if there is a reasonable way to build Clang
      // with its configuration macros set to establish these defaults rather
      // than doing it with runtime flags.
      "-rtlib=compiler-rt",
      "-unwindlib=libunwind",
      "-stdlib=libc++",

      // Override the default linker to use.
      "-fuse-ld=lld",
  });

  // Add target-specific flags.
  llvm::Triple triple(target_str);
  switch (triple.getOS()) {
    case llvm::Triple::Darwin:
    case llvm::Triple::MacOSX:
      // On macOS we need to set the sysroot to a viable SDK. Currently, this
      // hard codes the path to be the unversioned symlink. The prefix is also
      // hard coded in Homebrew and so this seems likely to work reasonably
      // well. Homebrew and I suspect the Xcode Clang both have this hard coded
      // at build time, so this seems reasonably safe but we can revisit if/when
      // needed.
      args.push_back(
          "--sysroot=/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk");

      // We also need to insist on a modern linker, otherwise the driver tries
      // too old and deprecated flags. The specific number here comes from an
      // inspection of the Clang driver source code to understand where features
      // were enabled, and this appears to be the latest version to control
      // driver behavior.
      //
      // TODO: We should replace this with use of `lld` eventually.
      args.push_back("-mlinker-version=705");
      break;

    default:
      break;
  }

  // Append our exact header search paths for the various parts of the C++
  // standard library headers as we don't build a single unified tree.
  for (const std::filesystem::path& runtime_path :
       {install_paths.libunwind_path(), install_paths.libcxx_path(),
        install_paths.libcxxabi_path()}) {
    args.push_back(
        llvm::formatv("-stdlib++-isystem{0}", runtime_path / "include").str());
  }
}

}  // namespace Carbon
